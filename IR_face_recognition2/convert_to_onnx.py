import torch

from models.mobilenet import MobileNetV2

NetWork = MobileNetV2()

torch.set_grad_enabled(False)


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

device = torch.cuda.current_device()
pretrained_dict = torch.load('./weights/best_liveness_mobilenetv2.pth.tar', map_location=lambda storage, loc: storage.cuda(device))
if "state_dict" in pretrained_dict.keys():
    pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
else:
    pretrained_dict = remove_prefix(pretrained_dict, 'module.')

NetWork.load_state_dict(pretrained_dict, strict=False)

NetWork.eval()
print('Finished loading model!')
print(NetWork)
device = torch.device("cuda")
NetWork = NetWork.to(device)

input_names = ["input"]
output_names = ["output"]
inputs = torch.randn(1, 3, 64, 64).to(device)
# torch_out = torch.onnx._export(net, inputs, output_onnx, export_params=True, verbose=False,
                               # input_names=input_names, output_names=output_names, keep_initializers_as_inputs=True)
torch_out = torch.onnx._export(NetWork, inputs, './weights/liveness_mobilenetv2.onnx', export_params=True,
                               input_names=input_names, output_names=output_names)
