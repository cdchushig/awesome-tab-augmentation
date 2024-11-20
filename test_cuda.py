import torch

def test_cuda():
    # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available!")
        
        # Get the number of GPUs
        num_gpus = torch.cuda.device_count()
        print(f"Number of CUDA GPUs: {num_gpus}")
        
        # Print details for each GPU
        for i in range(num_gpus):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        
        # Test tensor computation on GPU
        try:
            device = torch.device("cuda:0")  # Use the first GPU
            tensor = torch.tensor([1.0, 2.0, 3.0], device=device)
            
            x = torch.rand(3, 3).cuda()
            y = torch.rand(3, 3).cuda()
            print((x + y).cpu())
            
            print(f"Tensor on GPU: {tensor}")
            print(f"Tensor multiplied by 2: {tensor * 2}")
            print("CUDA GPU is working correctly!")
        except Exception as e:
            print(f"Failed to perform tensor computation on GPU: {e}")
    else:
        print("CUDA is not available on this system.")

if __name__ == "__main__":
    test_cuda()
