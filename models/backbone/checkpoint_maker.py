import torch

# Path to the .pth.tar file
checkpoint_path = '/glb/hou/pt.sgs/data/ml_ai_us/uspcjc/SimCLR/runs/May30_13-43-09_houcy1-n-gp193a03.americas.shell.com/checkpoint_1000.pth.tar'

try:
    # Attempt to load as a TorchScript model
    model = torch.jit.load(checkpoint_path, map_location=torch.device('cpu'))
    # Save the model as a .pth file
    output_path = '/glb/hou/pt.sgs/data/ml_ai_us/uspcjc/SimCLR/runs/May30_13-43-09_houcy1-n-gp193a03.americas.shell.com/model.pth'
    torch.jit.save(model, output_path)
    print(f"Model saved as {output_path}")
except RuntimeError as e:
    print(f"Failed to load with torch.jit.load(): {e}")
    print("Trying to load as a regular checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    # Extract the state dictionary if present
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    # Save the state dictionary as a .pth file
    torch.save(state_dict, output_path)
    print(f"State dictionary saved as {output_path}")
