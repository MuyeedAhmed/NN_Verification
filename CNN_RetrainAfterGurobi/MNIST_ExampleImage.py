import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# Load one sample from MNIST
mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
image, label = mnist[0]

# Plot the image
plt.imshow(image.squeeze(), cmap='gray')
# plt.title(f"Label: {label}")
plt.axis('off')
plt.savefig('Figures/mnist_sample.pdf', bbox_inches='tight')
# plt.show()
