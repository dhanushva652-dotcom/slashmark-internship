# AI-Powered Indoor Obstacle Avoidance

A cleaned-up PyTorch version of the provided AirSim reference project.  
It keeps the same idea of **camera-based perception + control**, but adds a safer runtime loop:

- learns a driving policy from the provided `dataset/data.csv`
- reads the paired images directly from `dataset/imgs.zip`
- adds a lightweight obstacle-risk heuristic for emergency slowing/stopping
- supports webcam, video file, or optional AirSim runtime

## Dataset layout expected

After unzipping the reference archive, the project should contain:

```text
dataset/
  data.csv
  imgs.zip
```

The code reads `imgs.zip` directly, so you do **not** need to manually extract all images.

## Train

```bash
python train.py --dataset-root path/to/UAV-indoor-obstacle-avoidance-based-on-AI-technique-master
```

This will create:

```text
checkpoints/policy_net.pt
checkpoints/metadata.json
```

## Run a demo

Webcam:

```bash
python run_realtime.py --source 0 --checkpoint checkpoints/policy_net.pt
```

Video file:

```bash
python run_realtime.py --source path/to/video.mp4 --checkpoint checkpoints/policy_net.pt
```

## Optional AirSim mode

If AirSim is installed and running, you can use:

```bash
python run_realtime.py --airsim --checkpoint checkpoints/policy_net.pt
```

## What the model does

The policy network predicts:

- forward speed
- steering angle

A safety layer then checks the scene for obstacle clutter and can:

- slow down
- bias the steering left/right
- stop in emergency conditions

This makes the system closer to a real indoor obstacle-avoidance stack.
