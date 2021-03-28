# Behavioral_Cloning

Here are the steps to run this project on local machine. I have tested them on Mac and Unbuntu.

1. Download the car simulator from [here](https://github.com/udacity/self-driving-car-sim#available-game-builds-precompiled-builds-of-the-simulator) in Term 1. Both version 1 and version 2 will work.

2. If you would like to change the design of the car simulator, you can make it happen by using Unity. Here is [the tutorial](https://github.com/udacity/self-driving-car-sim#available-game-builds-precompiled-builds-of-the-simulator). I find [this blog post](https://kaigo.medium.com/how-to-install-udacitys-self-driving-car-simulator-on-ubuntu-20-04-14331806d6dd) very helpful.

3. Create a conda environment and install dependencies following instructions in [this repo](https://github.com/udacity/CarND-Term1-Starter-Kit).

There might be some errors with keras versions. Just keep the version you installed in the evrionment and the version the same.

4. In debugging, my car didn't move in autonomous control mode. A **solution** to this issue is to downgrade the python-socketio to 4.2.1