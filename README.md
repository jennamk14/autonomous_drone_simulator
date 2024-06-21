Developed for the paper: [A Framework for Autonomic Computing for In Situ Imageomics](https://ieeexplore.ieee.org/abstract/document/10336017)

This tool allows users to test autonomous drone navigation policies in simulation,
based on real-world drone missions. 
See [getting_started.ipynb](https://github.com/jennamk14/autonomous_drone_simulator/blob/master/getting_started.ipynb) for detailed instructions on the provided scripts.

**Funding Sources**

The Imageomics Institute is supported by the National Science Foundation Award No. 2118240. The ICICLE project is funded by the National Science Foundation (NSF) under grant number OAC-2112606.

If you use this code, please consider citing the following paper:

@INPROCEEDINGS{10336017,
  author={Kline, Jenna and Stewart, Christopher and Berger-Wolf, Tanya and Ramirez, Michelle and Stevens, Samuel and Babu, Reshma Ramesh and Banerji, Namrata and Sheets, Alec and Balasubramaniam, Sowbaranika and Campolongo, Elizabeth and Thompson, Matthew and Stewart, Charles V. and Kholiavchenko, Maksim and Rubenstein, Daniel I. and Van Tiel, Nina and Miliko, Jackson},
  booktitle={2023 IEEE International Conference on Autonomic Computing and Self-Organizing Systems (ACSOS)}, 
  title={A Framework for Autonomic Computing for In Situ Imageomics}, 
  year={2023},
  volume={},
  number={},
  pages={11-16},
  abstract={In situ imageomics is a new approach to study ecological, biological and evolutionary systems wherein large image and video data sets are captured in the wild and machine learning methods are used to infer biological traits of individual organisms, animal social groups, species, and even whole ecosystems. Monitoring biological traits over large spaces and long periods of time could enable new, data-driven approaches to wildlife conservation, biodiversity, and sustainable ecosystem management. However, to accurately infer biological traits, machine learning methods for images require voluminous and high quality data. Adaptive, data-driven approaches are hamstrung by the speed at which data can be captured and processed. Camera traps and unmanned aerial vehicles (UAVs) produce voluminous data, but lose track of individuals over large areas, fail to capture social dynamics, and waste time and storage on images with poor lighting and view angles. In this vision paper, we make the case for a research agenda for in situ imageomics that depends on significant advances in autonomic and self-aware computing. Precisely, we seek autonomous data collection that manages camera angles, aircraft positioning, conflicting actions for multiple traits of interest, energy availability, and cost factors. Given the tools to detect object and identify individuals, we propose a research challenge: Which optimization model should the data collection system employ to accurately identify, characterize, and draw inferences from biological traits while respecting a budget? Using zebra and giraffe behavioral data collected over three weeks at the Mpala Research Centre in Laikipia County, Kenya, we quantify the volume and quality of data collected using existing approaches. Our proposed autonomic navigation policy for in situ imageomics collection has an F1 score of 82% compared to an expert pilot, and provides greater safety and consistency, suggesting great potential for state-of-the-art autonomic approaches if they can be scaled up to fully address the problem.},
  keywords={Social groups;Ecosystems;Wildlife;Machine learning;Data collection;Cameras;Object recognition;autonomous flight;UAVs;ecology;machine learning;computer vision;imageomics},
  doi={10.1109/ACSOS58161.2023.00018},
  ISSN={},
  month={Sep.},}

