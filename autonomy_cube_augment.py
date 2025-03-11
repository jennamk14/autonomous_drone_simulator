#!/usr/bin/env python3
"""
Drone Navigation Hypercube Generator

This script creates a hypercube from drone telemetry and video data, allowing for
virtual navigation through drone-captured imagery. The hypercube organizes frames
spatially and temporally, enabling testing of different navigation paths.

Usage:
    python drone_hypercube.py --data_dir <path_to_data> --video_file <video_prefix> [--custom_path <path_csv>]
"""

import os
import argparse
import time
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate and navigate drone imagery hypercube')
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Path to data directory')
    parser.add_argument('--video_file', type=str, required=True,
                        help='Video file prefix (e.g., DJI_0063)')
    parser.add_argument('--custom_path', type=str, default=None,
                        help='Path to CSV file with custom navigation path')
    parser.add_argument('--grid_size', type=float, default=0.00003,
                        help='Size of grid cells in degrees (default: 0.00003, approx. 3m)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save outputs (defaults to data_dir/video_file)')
    parser.add_argument('--visualize', action='store_true', 
                        help='Generate visualization of the drone path')
    parser.add_argument('--zoom_levels', type=int, default=2,
                        help='Number of zoom levels to process (1-3, default: 2)')
    parser.add_argument('--process_directions', type=str, default='all',
                        choices=['all', 'north', 'south', 'east', 'west', 'northeast', 'northwest', 'southeast', 'southwest'],
                        help='Which heading directions to fully process with alternative views')
    parser.add_argument('--side_views', action='store_true',
                        help='Include side views (left and right) for all directions')
    
    return parser.parse_args()


def ensure_directory(directory):
    """Ensure that a directory exists, creating it if necessary."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_image_dimensions(image_path):
    """Get dimensions of an image file."""
    im = Image.open(image_path)
    width, height = im.size
    return {
        'width': width,
        'height': height,
        'width_third': int(width/3),
        'width_sixth': int(width/6),
        'width_quarter': int(width/4),
        'height_quarter': int(height/4),
        'height_eighth': int(height/8)
    }


def load_flight_record(data_dir, video_file):
    """Load the flight record for the specified video."""
    flight_record_path = os.path.join(data_dir, video_file, f"{video_file}_flight_record.csv")
    
    if not os.path.exists(flight_record_path):
        print(f"Flight record not found: {flight_record_path}")
        exit(1)
        
    return pd.read_csv(flight_record_path)


def determine_heading_direction(heading_degrees):
    """Convert heading in degrees to cardinal direction."""
    directions = ['north', 'northeast', 'east', 'southeast', 
                 'south', 'southwest', 'west', 'northwest']
    
    # Normalize heading to 0-360
    heading_degrees = heading_degrees % 360
    
    # Calculate index (0-7) based on heading
    index = round(heading_degrees / 45) % 8
    
    return directions[index]


def preprocess_flight_data(flight_data):
    """Preprocess flight data, adding heading direction if not present."""
    # Make a copy to avoid modifying the original dataframe
    flight = flight_data.copy()
    
    # If heading_direction is not present, calculate it from gimbal_heading
    if 'heading_direction' not in flight.columns and 'gimbal_heading(degrees)' in flight.columns:
        flight['heading_direction'] = flight['gimbal_heading(degrees)'].apply(determine_heading_direction)
    
    return flight


def generate_grid(flight_data, grid_size=0.00003):
    """Generate a grid of latitude and longitude squares."""
    # Add a small buffer around the flight area
    fence = (
        flight_data['latitude'].min() - 0.001, 
        flight_data['longitude'].min() - 0.001,
        flight_data['latitude'].max() + 0.001, 
        flight_data['longitude'].max() + 0.001
    )
    
    # Divide fence area into grid_size x grid_size squares
    squares_latitude = np.arange(fence[0], fence[2], grid_size)
    squares_longitude = np.arange(fence[1], fence[3], grid_size)
    
    return squares_latitude, squares_longitude


def create_empty_cube(latitudes, longitudes, times, heading_directions):
    """Create an empty hypercube with the given dimensions."""
    # Create empty cube with placeholder frame reference
    frames = "frame_XXXXX.jpg"  # Placeholder
    
    # Create DataArray with coordinates
    cube = xr.DataArray(
        frames, 
        coords=[latitudes, longitudes, times, heading_directions],
        dims=['latitude', 'longitude', 'times', 'heading']
    )
    
    return cube


def process_frame(image, dims, output_path, image_index):
    """Process a frame, creating cropped versions for the hypercube."""
    # Center crop
    f = image[dims['height_quarter']: (dims['height_quarter']*3), 
              dims['width_third']: (dims['width_third']*2)]
    
    crop_path = os.path.join(output_path, f"crop_frame_{image_index}.jpg")
    cv2.imwrite(crop_path, f)
    
    return crop_path, image_index + 1


def process_directional_view(image, dims, output_path, image_index, cube, ilat, ilon, itime, iheading, 
                        direction, zoom_levels=2, side_views=True):
    """Process frames for any heading direction with variable zoom levels and side views."""
    # Base crop path from center
    base_path, image_index = process_frame(image, dims, output_path, image_index)
    cube[ilat, ilon, itime, iheading] = base_path
    
    # Set offsets based on heading direction
    # For grid positioning, we need to know which direction to place zoomed views
    lat_offset = 0
    lon_offset = 0
    
    if direction in ['north', 'northeast', 'northwest']:
        lon_offset = -1  # Zoom out goes "behind" (south of) the current position
    elif direction in ['south', 'southeast', 'southwest']:
        lon_offset = 1   # Zoom out goes "behind" (north of) the current position
    elif direction == 'east':
        lat_offset = -1  # Zoom out goes "behind" (west of) the current position
    elif direction == 'west':
        lat_offset = 1   # Zoom out goes "behind" (east of) the current position
    
    # Process zoom level 1 (zoomed out)
    if zoom_levels >= 1:
        # Zoom out 1
        bottom = int(dims['height_eighth'])
        top = int(dims['height_eighth']*7)
        left = int(dims['width_third'])
        right = int(dims['width_third']*2)
        
        f = image[bottom:top, left:right]
        crop_path = os.path.join(output_path, f"crop_frame_{image_index}.jpg")
        cv2.imwrite(crop_path, f)
        
        # Place in grid based on direction
        if lat_offset != 0:
            cube[ilat + lat_offset, ilon, itime, iheading] = crop_path
        else:
            cube[ilat, ilon + lon_offset, itime, iheading] = crop_path
        image_index += 1
        
        # Process side views for zoom level 1
        if side_views:
            # Left side view
            f = image[bottom:top, int(left - dims['width_sixth']): int(left + dims['width_sixth'])]
            crop_path = os.path.join(output_path, f"crop_frame_{image_index}.jpg")
            cv2.imwrite(crop_path, f)
            
            # Place left view
            if lat_offset != 0:
                cube[ilat + lat_offset - 1, ilon, itime, iheading] = crop_path
            else:
                cube[ilat - 1, ilon + lon_offset, itime, iheading] = crop_path
            image_index += 1
            
            # Right side view
            f = image[bottom:top, int(right - dims['width_sixth']): int(right + dims['width_sixth'])]
            crop_path = os.path.join(output_path, f"crop_frame_{image_index}.jpg")
            cv2.imwrite(crop_path, f)
            
            # Place right view
            if lat_offset != 0:
                cube[ilat + lat_offset + 1, ilon, itime, iheading] = crop_path
            else:
                cube[ilat + 1, ilon + lon_offset, itime, iheading] = crop_path
            image_index += 1
    
    # Process zoom level 2 (more zoomed out)
    if zoom_levels >= 2:
        # Zoom out 2
        bottom = int(dims['height_eighth'])
        top = int(dims['height_eighth']*7)
        left = int(dims['width_quarter'])
        right = int(dims['width_quarter']*3)
        
        f = image[bottom:top, left:right]
        crop_path = os.path.join(output_path, f"crop_frame_{image_index}.jpg")
        cv2.imwrite(crop_path, f)
        
        # Place in grid based on direction
        if lat_offset != 0:
            cube[ilat + (lat_offset*2), ilon, itime, iheading] = crop_path
        else:
            cube[ilat, ilon + (lon_offset*2), itime, iheading] = crop_path
        image_index += 1
        
        # Process side views for zoom level 2
        if side_views:
            # Left side view
            f = image[bottom:top, int(left - dims['width_sixth']): int(left + dims['width_sixth'])]
            crop_path = os.path.join(output_path, f"crop_frame_{image_index}.jpg")
            cv2.imwrite(crop_path, f)
            
            # Place left view
            if lat_offset != 0:
                cube[ilat + (lat_offset*2) - 1, ilon, itime, iheading] = crop_path
            else:
                cube[ilat - 1, ilon + (lon_offset*2), itime, iheading] = crop_path
            image_index += 1
            
            # Right side view
            f = image[bottom:top, int(right - dims['width_sixth']): int(right + dims['width_sixth'])]
            crop_path = os.path.join(output_path, f"crop_frame_{image_index}.jpg")
            cv2.imwrite(crop_path, f)
            
            # Place right view
            if lat_offset != 0:
                cube[ilat + (lat_offset*2) + 1, ilon, itime, iheading] = crop_path
            else:
                cube[ilat + 1, ilon + (lon_offset*2), itime, iheading] = crop_path
            image_index += 1
    
    # Process zoom level 3 (zoomed in)
    if zoom_levels >= 3:
        # Zoom in
        bottom = int(dims['height_quarter']*1.5)
        top = int(dims['height_quarter']*2.5)
        left = int(dims['width_third']*1.25)
        right = int(dims['width_third']*1.75)
        
        f = image[bottom:top, left:right]
        crop_path = os.path.join(output_path, f"crop_frame_{image_index}.jpg")
        cv2.imwrite(crop_path, f)
        
        # Place in grid based on direction (opposite of zoom out)
        if lat_offset != 0:
            cube[ilat - lat_offset, ilon, itime, iheading] = crop_path
        else:
            cube[ilat, ilon - lon_offset, itime, iheading] = crop_path
        image_index += 1
        
        # Process side views for zoom in
        if side_views:
            # Left side view
            f = image[bottom:top, int(left - dims['width_sixth']/2): int(left + dims['width_sixth']/2)]
            crop_path = os.path.join(output_path, f"crop_frame_{image_index}.jpg")
            cv2.imwrite(crop_path, f)
            
            # Place left view
            if lat_offset != 0:
                cube[ilat - lat_offset - 1, ilon, itime, iheading] = crop_path
            else:
                cube[ilat - 1, ilon - lon_offset, itime, iheading] = crop_path
            image_index += 1
            
            # Right side view
            f = image[bottom:top, int(right - dims['width_sixth']/2): int(right + dims['width_sixth']/2)]
            crop_path = os.path.join(output_path, f"crop_frame_{image_index}.jpg")
            cv2.imwrite(crop_path, f)
            
            # Place right view
            if lat_offset != 0:
                cube[ilat - lat_offset + 1, ilon, itime, iheading] = crop_path
            else:
                cube[ilat + 1, ilon - lon_offset, itime, iheading] = crop_path
            image_index += 1
    
    return cube, image_index


def build_hypercube(flight_data, dimensions, output_dir, grid_size=0.00003, 
                 zoom_levels=2, process_directions='all', side_views=True):
    """Build a hypercube from flight data and video frames with customizable processing options."""
    print("Building hypercube...")
    
    # Create cropped frames directory if it doesn't exist
    cropped_frames_dir = os.path.join(output_dir, "cropped_frames")
    ensure_directory(cropped_frames_dir)
    
    # Preprocess flight data
    flight = preprocess_flight_data(flight_data)
    
    # Generate grid
    squares_latitude, squares_longitude = generate_grid(flight, grid_size)
    print(f"Grid size: {len(squares_latitude)} latitude x {len(squares_longitude)} longitude squares")
    print(f"Grid cell size: {grid_size} degrees (approximately {grid_size * 111000:.1f} meters)")
    
    # Define coordinate variables
    times = np.arange(0, len(flight), 1)
    heading_directions = ['north', 'northeast', 'east', 'southeast', 
                          'south', 'southwest', 'west', 'northwest']
    
    # Create empty cube
    cube = create_empty_cube(squares_latitude, squares_longitude, times, heading_directions)
    
    # Determine which directions to fully process
    if process_directions == 'all':
        directions_to_process = heading_directions
    else:
        directions_to_process = [process_directions]
    
    print(f"Processing directions: {', '.join(directions_to_process)}")
    print(f"Zoom levels: {zoom_levels}")
    print(f"Including side views: {side_views}")
    
    # Fill the cube with frame data
    path = []
    image_index = 0
    
    for idx, row in flight.iterrows():
        t = idx
        lat, lon = row['latitude'], row['longitude']
        
        if 'heading_direction' in row:
            head = row['heading_direction']
        else:
            head = determine_heading_direction(row['gimbal_heading(degrees)'])
        
        # Find nearest grid indices
        ilat = list(cube.latitude.values).index(cube.sel(latitude=lat, method='nearest').latitude)
        ilon = list(cube.longitude.values).index(cube.sel(longitude=lon, method='nearest').longitude)
        itime = list(cube.times.values).index(cube.sel(times=t).times)
        
        try:
            iheading = list(cube.heading.values).index(cube.sel(heading=head).heading)
        except ValueError:
            print(f"Warning: Heading '{head}' not found in cube headings. Using 'north' as default.")
            head = 'north'
            iheading = list(cube.heading.values).index(cube.sel(heading=head).heading)
        
        # Process frame
        try:
            image = cv2.imread(row['frame'])
            
            if image is None:
                print(f"Warning: Could not read image {row['frame']}")
                continue
                
            # Basic processing for all headings - center view
            crop_path, image_index = process_frame(image, dimensions, cropped_frames_dir, image_index)
            cube[ilat, ilon, itime, iheading] = crop_path
            
            # Additional processing for specified headings
            if head in directions_to_process:
                cube, image_index = process_directional_view(
                    image, dimensions, cropped_frames_dir, image_index, 
                    cube, ilat, ilon, itime, iheading, head, 
                    zoom_levels=zoom_levels, side_views=side_views
                )
                
            # Record path
            path.append([ilat, ilon, itime, iheading])
            
            # Print progress every 10 frames
            if idx % 10 == 0:
                print(f"Processed {idx}/{len(flight)} frames")
            
        except Exception as e:
            print(f"Error processing frame {row['frame']}: {e}")
    
    print(f"Hypercube built with {image_index} processed frames")
    return cube, path


def save_hypercube(cube, output_dir, video_file):
    """Save the hypercube as a netCDF file."""
    output_path = os.path.join(output_dir, f"{video_file}_cube.nc")
    cube.to_netcdf(output_path)
    print(f"Hypercube saved to: {output_path}")
    return output_path


def visualize_path(flight_data, path, output_dir, video_file):
    """Generate a visualization of the drone path."""
    plt.figure(figsize=(12, 10))
    
    # Plot the actual flight path
    plt.plot(flight_data['longitude'], flight_data['latitude'], 'b-', label='Actual Flight Path')
    
    # Mark start and end
    plt.plot(flight_data['longitude'].iloc[0], flight_data['latitude'].iloc[0], 'go', markersize=10, label='Start')
    plt.plot(flight_data['longitude'].iloc[-1], flight_data['latitude'].iloc[-1], 'ro', markersize=10, label='End')
    
    # Add arrows to show direction
    arrow_indices = np.linspace(0, len(flight_data)-1, 20, dtype=int)
    for i in arrow_indices:
        if i+1 < len(flight_data):
            plt.arrow(
                flight_data['longitude'].iloc[i], 
                flight_data['latitude'].iloc[i],
                (flight_data['longitude'].iloc[i+1] - flight_data['longitude'].iloc[i])/2,
                (flight_data['latitude'].iloc[i+1] - flight_data['latitude'].iloc[i])/2,
                head_width=0.00002, head_length=0.00002, fc='b', ec='b'
            )
    
    plt.title('Drone Flight Path')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    plt.legend()
    
    # Save figure
    output_path = os.path.join(output_dir, f"{video_file}_path.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Path visualization saved to: {output_path}")


def load_custom_path(path_file):
    """Load a custom navigation path from a CSV file."""
    if not os.path.exists(path_file):
        print(f"Custom path file not found: {path_file}")
        return None
    
    try:
        custom_path = pd.read_csv(path_file)
        required_columns = ['latitude', 'longitude', 'heading_direction']
        
        if not all(col in custom_path.columns for col in required_columns):
            print(f"Custom path file must contain columns: {required_columns}")
            return None
            
        return custom_path
    except Exception as e:
        print(f"Error loading custom path: {e}")
        return None


def test_custom_path(cube, custom_path):
    """Test a custom navigation path through the hypercube with additional options."""
    results = []
    
    for idx, row in custom_path.iterrows():
        lat, lon = row['latitude'], row['longitude']
        head = row['heading_direction']
        
        # Handle additional navigation parameters if present
        zoom_level = row.get('zoom_level', 0)  # 0 = normal, -1/-2 = zoomed out, +1 = zoomed in
        side_view = row.get('side_view', 'center')  # 'center', 'left', or 'right'
        
        # Find nearest values in the cube
        nearest_lat = cube.sel(latitude=lat, method='nearest').latitude.item()
        nearest_lon = cube.sel(longitude=lon, method='nearest').longitude.item()
        
        # Use the first timestamp for simplicity (or can use a specific timestamp if needed)
        time_value = cube.times.values[0]
        
        # Adjust grid location based on navigation parameters
        grid_lat = nearest_lat
        grid_lon = nearest_lon
        
        # Apply zoom level offset
        if zoom_level != 0:
            # North/South
            if head in ['north', 'northeast', 'northwest']:
                grid_lon -= zoom_level  # Negative zoom level moves backwards (south


def main():
    """Main function to run the script."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(args.data_dir, args.video_file)
    
    ensure_directory(output_dir)
    ensure_directory(os.path.join(output_dir, "cropped_frames"))
    
    print(f"Processing video: {args.video_file}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Grid size: {args.grid_size} degrees")
    
    # Start timing
    start_time = time.time()
    
    # Load flight record
    flight_data = load_flight_record(args.data_dir, args.video_file)
    
    # Find at least one frame to get dimensions
    sample_frames = [row['frame'] for idx, row in flight_data.iterrows() 
                    if os.path.exists(row['frame'])]
    
    if not sample_frames:
        print("No valid frames found in flight record.")
        exit(1)
    
    # Get image dimensions
    dimensions = get_image_dimensions(sample_frames[0])
    print(f"Image dimensions: {dimensions['width']}x{dimensions['height']} pixels")
    
    # Build hypercube with enhanced options
    cube, path = build_hypercube(
        flight_data, 
        dimensions, 
        output_dir, 
        grid_size=args.grid_size,
        zoom_levels=args.zoom_levels,
        process_directions=args.process_directions,
        side_views=args.side_views
    )
    
    # Save the hypercube
    cube_path = save_hypercube(cube, output_dir, args.video_file)
    
    # Visualize path if requested
    if args.visualize:
        visualize_path(flight_data, path, output_dir, args.video_file)
    
    # Test custom path if provided
    if args.custom_path:
        custom_path_data = load_custom_path(args.custom_path)
        if custom_path_data is not None:
            print(f"Testing custom path from: {args.custom_path}")
            results = test_custom_path(cube, custom_path_data)
            
            # Save results
            results_path = os.path.join(output_dir, f"{args.video_file}_custom_path_results.csv")
            results.to_csv(results_path, index=False)
            print(f"Custom path results saved to: {results_path}")
            
            # Create a simple HTML viewer for the custom path results
            html_path = os.path.join(output_dir, f"{args.video_file}_custom_path_viewer.html")
            with open(html_path, 'w') as f:
                f.write(f"""<!DOCTYPE html>
                <html>
                <head>
                    <title>Custom Path Viewer - {args.video_file}</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        h1 {{ color: #2c3e50; }}
                        .frame-container {{ display: flex; flex-wrap: wrap; gap: 20px; margin-top: 20px; }}
                        .frame-item {{ border: 1px solid #ddd; padding: 10px; border-radius: 5px; }}
                        .frame-item img {{ max-width: 400px; max-height: 300px; }}
                        .frame-info {{ margin-top: 10px; }}
                    </style>
                </head>
                <body>
                    <h1>Custom Navigation Path - {args.video_file}</h1>
                    <p>This page shows the frames retrieved for the custom navigation path.</p>
                    <div class="frame-container">
                """)
                                
                                for i, row in enumerate(results.iterrows()):
                                    row_data = row[1]
                                    f.write(f"""
                        <div class="frame-item">
                            <h3>Waypoint {i+1}</h3>
                            <img src="{row_data['frame_path']}" alt="Frame at waypoint {i+1}">
                            <div class="frame-info">
                                <p><strong>Location:</strong> {row_data['nearest_lat']:.6f}, {row_data['nearest_lon']:.6f}</p>
                                <p><strong>Heading:</strong> {row_data['heading']}</p>
                            </div>
                        </div>
                """)
                                
                                f.write("""
                    </div>
                </body>
                </html>""")
            
            print(f"Custom path viewer created at: {html_path}")
    
    # End timing
    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")
    print(f"Hypercube successfully created at: {cube_path}")
    print("\nYou can now use this hypercube to explore different drone navigation paths.")


if __name__ == "__main__":
    main()