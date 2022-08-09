import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math

from PIL import Image
from einops import rearrange
import utils


########################### 1. EAST model ##########################################
class VGG(nn.Module):
	def __init__(self, features):
		super(VGG, self).__init__()
		self.features = features
		self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
		self.classifier = nn.Sequential(
			nn.Linear(512 * 7 * 7, 4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096, 1000),
		)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		x = self.features(x)
		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		print(x)
		return x


class extractor(nn.Module):
	def __init__(self, pretrained):
		super(extractor, self).__init__()
		vgg16_bn = VGG(utils.make_layers(utils.OCR_VGG_cfg, batch_norm=True))
		if pretrained:
			vgg16_bn.load_state_dict(torch.load('./pths/vgg16_bn-6c64b313.pth'))
		self.features = vgg16_bn.features
	
	def forward(self, x):
		out = []
		for m in self.features:
			x = m(x)
			if isinstance(m, nn.MaxPool2d):
				out.append(x)
		return out[1:]


class merge(nn.Module):
	def __init__(self):
		super(merge, self).__init__()

		self.conv1 = nn.Conv2d(1024, 128, 1)
		self.bn1 = nn.BatchNorm2d(128)
		self.relu1 = nn.ReLU()
		self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
		self.bn2 = nn.BatchNorm2d(128)
		self.relu2 = nn.ReLU()

		self.conv3 = nn.Conv2d(384, 64, 1)
		self.bn3 = nn.BatchNorm2d(64)
		self.relu3 = nn.ReLU()
		self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
		self.bn4 = nn.BatchNorm2d(64)
		self.relu4 = nn.ReLU()

		self.conv5 = nn.Conv2d(192, 32, 1)
		self.bn5 = nn.BatchNorm2d(32)
		self.relu5 = nn.ReLU()
		self.conv6 = nn.Conv2d(32, 32, 3, padding=1)
		self.bn6 = nn.BatchNorm2d(32)
		self.relu6 = nn.ReLU()

		self.conv7 = nn.Conv2d(32, 32, 3, padding=1)
		self.bn7 = nn.BatchNorm2d(32)
		self.relu7 = nn.ReLU()
		
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		y = F.interpolate(x[3], scale_factor=2, mode='bilinear', align_corners=True)
		y = torch.cat((y, x[2]), 1)
		y = self.relu1(self.bn1(self.conv1(y)))		
		y = self.relu2(self.bn2(self.conv2(y)))
		
		y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
		y = torch.cat((y, x[1]), 1)
		y = self.relu3(self.bn3(self.conv3(y)))		
		y = self.relu4(self.bn4(self.conv4(y)))
		
		y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
		y = torch.cat((y, x[0]), 1)
		y = self.relu5(self.bn5(self.conv5(y)))		
		y = self.relu6(self.bn6(self.conv6(y)))
		
		y = self.relu7(self.bn7(self.conv7(y)))
		return y

class output(nn.Module):
	def __init__(self, scope=512):
		super(output, self).__init__()
		self.conv1 = nn.Conv2d(32, 1, 1)
		self.sigmoid1 = nn.Sigmoid()
		self.conv2 = nn.Conv2d(32, 4, 1)
		self.sigmoid2 = nn.Sigmoid()
		self.conv3 = nn.Conv2d(32, 1, 1)
		self.sigmoid3 = nn.Sigmoid()
		self.scope = 512
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)

	def forward(self, x):
		score = self.sigmoid1(self.conv1(x))
		loc   = self.sigmoid2(self.conv2(x)) * self.scope
		angle = (self.sigmoid3(self.conv3(x)) - 0.5) * math.pi
		geo   = torch.cat((loc, angle), 1) 
		return score, geo
		
	
class EAST(nn.Module):
	def __init__(self, pretrained=True):
		super(EAST, self).__init__()
		self.extractor = extractor(pretrained)
		self.merge     = merge()
		self.output    = output()
	
	def forward(self, x):
		return self.output(self.merge(self.extractor(x)))


################################## 2. Position, text, image embeding ###################################
def image_crop(img,box):
	cropped_img = img.crop(box)
	return cropped_img


class ImageEmbedding(nn.Module):
	def __init__(self,features) -> None:
		super(ImageEmbedding,self).__init__()
		self.features = features
		self.avgpool = nn.AdaptiveAvgPool2d((3, 15))
		self.classifier = nn.Sequential(
			nn.Linear(utils.img_emb_cfg[-1] * 3 * 15, 1012),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(1012, 256),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(256, 60),
		)

	def forward(self, img, boxes):
		emb = []
		batch, c, image_h, image_w = img.shape
		for b in range(batch):
			batch_emb = []
			for box in boxes[b]:
				x_min,y_min,_,_,x_max,y_max,_,_,_ = box
				cropped_img = img[b][:,int(y_min):int(y_max),int(x_min):int(x_max)]
				cropped_img = self.features(cropped_img.unsqueeze(0))
				cropped_img = self.avgpool(cropped_img)
				cropped_img = cropped_img.view(cropped_img.size(0), -1)
				cropped_img = self.classifier(cropped_img)
				pos_embedding = torch.tensor([x_min/image_w,y_min/image_h,x_max/image_w,y_max/image_h])
				batch_emb.append(torch.cat((pos_embedding.unsqueeze(0),cropped_img),axis=1))
			emb.append(torch.cat(batch_emb,axis=0).unsqueeze(0))
		return torch.cat(emb,axis=0)


##################### Classification ###########################

class LayoutClassification(nn.Module):
	def __init__(self, emb_dim, out_dim , num_heads, dropout_ratio: float = 0.2, **kwargs):
		super().__init__()
		self.emb_block = ImageEmbedding(utils.make_layers(utils.img_emb_cfg, batch_norm=True))

		self.emb_dim = emb_dim
		self.num_heads = num_heads 
		self.scaling = (self.emb_dim // num_heads) ** -0.5
		
		self.value = nn.Linear(emb_dim, emb_dim)
		self.key = nn.Linear(emb_dim, emb_dim)
		self.query = nn.Linear(emb_dim, emb_dim)
		self.att_drop = nn.Dropout(dropout_ratio)

		self.linear = nn.Linear(emb_dim, out_dim)
				
	def forward(self, img, boxes):
		emb = self.emb_block(img,boxes)
		# query, key, value
		Q = self.query(emb)
		K = self.key(emb)
		V = self.value(emb)

		# q = k = v = patch_size**2 + 1 & h * d = emb_dim
		Q = rearrange(Q, 'b q (h d) -> b h q d', h=self.num_heads)
		K = rearrange(K, 'b k (h d) -> b h d k', h=self.num_heads)
		V = rearrange(V, 'b v (h d) -> b h v d', h=self.num_heads)

		## scaled dot-product
		weight = torch.matmul(Q, K) 
		weight = weight * self.scaling
		
		attention = torch.softmax(weight, dim=-1)
		attention = self.att_drop(attention)

		context = torch.matmul(attention, V) 
		context = rearrange(context, 'b h q d -> b q (h d)')

		x = self.linear(context)
		return x


if __name__ == '__main__':
	img_path    = '../FUNSD/testing_data/images/82092117.png'
	model_path  = './pths/east_vgg16.pth'
	ocr_result     = './out/ocr_result.png'
	label = ["other","header","question","answer"]
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	model = EAST().to(device)
	model.load_state_dict(torch.load(model_path))
	model.eval()
	img = Image.open(img_path).convert('RGB')
	img_size = img.size
	boxes = utils.detect(img, model, device)
	
	model = LayoutClassification(64,5,8)
	input = utils.load_pil(img).unsqueeze(0)
	print('input img shape:',input.shape)
	input_box = boxes.unsqueeze(0)
	print('input boxes shape',input_box.shape)
	out = model(input,input_box)
	print('output shape:',out.shape)

	utils.plot_boxes(img, boxes)
	img.save(ocr_result)