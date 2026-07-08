import torch
import torchvision
import torchvision.transforms as transforms
import struct
import numpy as np
from torch.utils.data import DataLoader

def export_mnist_float16(model):
    # Set model to evaluation and convert entire network architecture to Float16
    model.eval()
    model = model.half() 
    print("Model structural parameters converted to Float16.")
    
    # 1. Prepare MNIST data (Resize to 32x32, duplicate Grayscale to 3-channel RGB)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)) # 1x32x32 -> 3x32x32
    ])
    
    # Load validation loop data
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=100, shuffle=False)

    # 2. EXPORT WEIGHTS & BIASES (As raw Float16 half-precision fields)
    print("Writing weights to model_weights_float16.bin...")
    with torch.no_grad():
        with open("model_weights_float16.bin", "wb") as f:
            for name, param in model.named_parameters():
                # Extract float16 array representation directly
                np_fp16 = param.cpu().numpy().astype(np.float16)
                f.write(np_fp16.tobytes())
                
    print("-> Saved model_weights_float16.bin")

    # 3. EXPORT TEST DATA & PYTORCH REFERENCE PREDICTIONS
    print("Processing and exporting full 10,000 image MNIST validation loop...")
    total_samples = len(test_loader.dataset)
    pytorch_predictions = []

    with open("test_data_float16.bin", "wb") as fd:
        fd.write(struct.pack("i", total_samples)) # Header record
        
        with torch.no_grad():
            for images, labels in test_loader:
                # Convert input batch to float16 to match reference execution exactly
                images_fp16 = images.half()
                
                # Write individual image bytes to file
                for i in range(images.size(0)):
                    single_img = images_fp16[i].numpy().astype(np.float16)
                    true_label = int(labels[i])
                    
                    fd.write(single_img.tobytes())
                    fd.write(struct.pack("i", true_label))
                
                # Generate native PyTorch Float16 inference predictions
                outputs = model(images_fp16)
                preds = outputs.argmax(dim=1).cpu().numpy()
                pytorch_predictions.extend(preds)

    print(f"-> Saved test_data_float16.bin ({total_samples} samples)")

    # Save reference targets to array
    with open("pytorch_preds_float16.bin", "wb") as fp:
        fp.write(np.array(pytorch_predictions, dtype=np.int32).tobytes())
        fp.flush()
        
    print("-> Saved pytorch_preds_float16.bin")
    print("Float16 Export complete!")

if __name__ == "__main__":
    from torchvision.models import shufflenet_v2_x0_5
    my_model = shufflenet_v2_x0_5(num_classes=10)
    
    # If you have your trained weights, load them here before converting to half-precision:
    # my_model.load_state_dict(torch.load("mnist_shufflenet.pth", map_location="cpu"))
    
    export_mnist_float16(my_model)