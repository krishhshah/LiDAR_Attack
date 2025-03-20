# OpenPCDet Attack and Visualization

This repository provides tools to generate and visualize adversarial point clouds for OpenPCDet, specifically targeting the SECOND model trained on the KITTI dataset.

## Setting Up

1.  **OpenPCDet Setup:**
    * Follow the instructions in the OpenPCDet `README.md` located in `docs/` to set up OpenPCDet using the KITTI dataset and the SECOND model.
    * This process will require downloading `.bin` files, which will be stored under `/data/kitti`. We will refer to this path as `$KITTI_BIN_DATA_PATH`.
    * Ensure the initial OpenPCDet setup is fully functional before proceeding.

2.  **Docker Setup (Specific to Nader Sehatbaksh's Server):**
    * **Note:** These steps were used on Nader Sehatbaksh's server. Adaptations may be necessary for other environments.
    * Navigate to the `docker` directory:
        ```bash
        cd docker
        ```
    * Build the Docker image:
        ```bash
        sudo docker build -f 12.0_Dockerfile -t openpcdet-docker12 .
        ```
    * Run the Docker container with GPU support and volume mounts:
        ```bash
        sudo docker run -it --net=host --gpus 'all,"capabilities=compute,utility,graphics"' -v /tmp/.X11-unix:/tmp/.X11-unix -v "$PWD":/root -e DISPLAY -v $KITTI_BIN_DATA_PATH:/LiDAR_Attack/data openpcdet-docker12 bash
        ```
        * `--net=host`: Enables host networking.
        * `--gpus 'all,"capabilities=compute,utility,graphics"'`: Enables GPU support.
        * `-v /tmp/.X11-unix:/tmp/.X11-unix`: Mounts the X11 socket for visualization.
        * `-v "$PWD":/root`: Mounts the current directory into the Docker container.
        * `-e DISPLAY`: Sets the display environment variable.
        * `-v $KITTI_BIN_DATA_PATH:/LiDAR_Attack/data`: Mounts the KITTI `.bin` data into the container at `/LiDAR_Attack/data`.

## Generating Point Clouds

Within the Docker container, use the following commands to generate adversarial point clouds:

* **Generate Attack Point Clouds:**
    ```bash
    python3 attack.py --cfg_file cfgs/kitti_models/second.yaml --ckpt second.pth --save_to_file --save_points --eps 0.1
    ```
    * `--cfg_file`: Specifies the configuration file for the SECOND model.
    * `--ckpt`: Specifies the path to the trained SECOND model checkpoint.
    * `--save_to_file`: Saves the generated point clouds to files.
    * `--save_points`: Saves the point cloud points.
    * `--eps 0.1`: Sets the perturbation magnitude (epsilon) for the attack.

* **Generate Attack + Defense Point Clouds:**
    ```bash
    python3 attack.py --cfg_file cfgs/kitti_models/second.yaml --ckpt second.pth --save_to_file --save_points --eps 0.1 --defense
    ```
    * `--defense`: Enables the defense mechanism during attack generation.

## Visualizing Point Clouds

After generating the point clouds, you can visualize them using the following methods:

1.  **Visualize with Labels (using `visualize.py`):**
    * Navigate to the `visualize` directory:
        ```bash
        cd visualize
        ```
    * Run the visualization script:
        ```bash
        python3 visualize.py --data=$POINT_CLOUD_DATA_PATH --frame=$FRAME_NUMBER
        ```
        * `$POINT_CLOUD_DATA_PATH`: Path to the directory containing the point cloud data.
        * `$FRAME_NUMBER`: The frame number to visualize.

2.  **Visualize using OpenPCDet Demo (using `demo.py`):**
    * Navigate to the `tools` directory:
        ```bash
        cd tools
        ```
    * Run the OpenPCDet demo script:
        ```bash
        python3 demo.py --cfg_file=cfgs/kitti_models/second.yaml --ckpt=../second.pth --data_path=$POINT_CLOUD_DATA_PATH
        ```
        * `--cfg_file`: Specifies the configuration file for the SECOND model.
        * `--ckpt`: Specifies the path to the trained SECOND model checkpoint.
        * `--data_path`: Specifies the path to the directory containing the point cloud data.
