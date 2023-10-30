'''
VEHICLES:这取决于你感兴趣的车辆类型,例如,可以设为 ["Car", "Van", "Truck"]。

BIN:可能的值可能在10到90之间,这取决于你需要多精细的方向划分。一个常用的值是30,即每个方向区间为12度。

OVERLAP:这通常需要设置为0到1之间的一个较小值。可能的值可以是0.1。

MAX_JIT:这取决于你希望有多大的随机边界框偏移。如果你的图像大小是224,那么可能的值可以是30。

NORM_H, NORM_W:这需要根据你的模型的输入要求进行设置。常见的值可能是224,这是许多卷积神经网络的默认输入大小。

label_dir, image_dir:这需要设置为你的标签文件和图像文件的实际存储路径。

batch_size:这需要根据你的硬件资源和模型大小进行设置。一个常见的值可能是32,但如果你的模型非常大或者你的硬件资源有限,你可能需要选择一个更小的值。
'''

BIN, OVERLAP = 30, 0.1
MAX_JIT = 30
NORM_H, NORM_W = 224, 224
VEHICLES = ['Car', 'Van', 'Truck', 'Pedestrian', 'Sitter', 'Cyclist', 'Tram', 'Misc']
BATCH_SIZE = 8

label_dir = 'E:/Codespace/image-to-3d-bbox/data/image_to_train/data_object_label_2/training/label_2/'
image_dir = 'E:/Codespace/image-to-3d-bbox/data/image_to_train/data_object_image_2/training/image_2/'