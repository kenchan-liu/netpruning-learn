import torch
from torch import nn
from model import  *
dummy_input=torch.randn(10,3,256,256,device='cuda')
ckpt = torch.load("f://GNR_checkpoint_full.pt", map_location=lambda storage, loc: storage)
device='cuda'
G_A2B = Generator( 256, 3, 8, 5, lr_mlp=0.01, n_res=1).to(device)
G_A2B = nn.DataParallel(G_A2B).cuda()

D_A = Discriminator(256).to(device)
G_B2A = Generator( 256, 3, 8, 5, lr_mlp=0.01, n_res=1).to(device)
G_B2A = nn.DataParallel(G_B2A).cuda()

D_B = Discriminator(256).to(device)
D_L = LatDiscriminator(8).to(device)
G_A2B.load_state_dict(ckpt['G_A2B'],strict=False)
G_B2A.load_state_dict(ckpt['G_B2A'],strict=False)
D_A.load_state_dict(ckpt['D_A'])
D_B.load_state_dict(ckpt['D_B'])
D_L.load_state_dict(ckpt['D_L'])

print(torch.cuda.is_available())
if __name__=="__main__":
    input_names = ["input"]
    output_names = ["output"]

    torch.onnx.export(G_A2B,
                  dummy_input,
                  "G_A2B.onnx",
                  verbose=True,
                  input_names=input_names,
                  output_names=output_names)

