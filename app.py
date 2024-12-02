import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import json
from urllib.request import urlopen
import matplotlib.pyplot as plt
import numpy as np
import random

class AdversarialNoiseGenerator:
    def __init__(self, model=None, device='cpu'):
        """
        Initialize the AdversarialNoiseGenerator.

        Parameters:
        - model: A pre-trained torchvision model. If None, ResNet50 is used.
        - device: 'cpu' or 'cuda' for GPU acceleration.
        """
        if model is None:
            self.model = models.resnet50(pretrained=True)
        else:
            self.model = model
        self.model.eval()
        self.device = device
        self.model.to(self.device)

        # Preprocessing transforms
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

        # Normalization values for ImageNet
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(3,1,1).to(self.device)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).view(3,1,1).to(self.device)

        # Load ImageNet class labels
        class_idx = json.load(urlopen(
            "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"))
        self.idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
        self.label2idx = {class_idx[str(k)][1]: int(k) for k in range(len(class_idx))}
        
    def get_class_name(self, class_index):
        """
        Get the class name for a given class index.

        Parameters:
        - class_index: The index of the class.

        Returns:
        - Class name corresponding to the class index.
        """
        if 0 <= class_index < len(self.idx2label):
            return self.idx2label[class_index]
        else:
            raise ValueError(f"Class index '{class_index}' is out of range.")
        
    def get_class_index(self, class_name):
        """
        Get the class index for a given class name.

        Parameters:
        - class_name: The name of the class.

        Returns:
        - Class index corresponding to the class name.
        """
        if class_name in self.label2idx:
            return self.label2idx[class_name]
        else:
            raise ValueError(f"Class name '{class_name}' not found in ImageNet classes.")

    def get_random_class(self, exclude_class=None):
        """
        Select a random class index from ImageNet classes.

        Parameters:
        - exclude_class: Class index or name to exclude from selection.

        Returns:
        - Tuple of (class index, class name).
        """
        if exclude_class is not None:
            if isinstance(exclude_class, str):
                exclude_class = self.get_class_index(exclude_class)
            elif not isinstance(exclude_class, int):
                raise ValueError("exclude_class must be a string (class name) or an integer (class index).")
            available_classes = list(range(len(self.idx2label)))
            available_classes.remove(exclude_class)
        else:
            available_classes = list(range(len(self.idx2label)))
        target_idx = random.choice(available_classes)
        target_name = self.idx2label[target_idx]
        return target_idx, target_name

    def targeted_attack(self, image, target_label, epsilon, num_iterations):
        """
        Perform a targeted adversarial attack.

        Parameters:
        - image: Normalized input image tensor (1 x C x H x W).
        - target_label: Target class index.
        - epsilon: Step size for each iteration.
        - num_iterations: Number of iterations to run the attack.

        Returns:
        - Adversarial image tensor.
        """
        adv_image = image.clone().detach().requires_grad_(True)

        criterion = nn.CrossEntropyLoss()

        for _ in range(num_iterations):
            outputs = self.model(adv_image)
            loss = criterion(outputs, torch.LongTensor([target_label]).to(self.device))

            self.model.zero_grad()
            loss.backward()

            # For targeted attack, subtract the gradient
            adv_image.data = adv_image.data - epsilon * adv_image.grad.data.sign()

            # Clamp the image to maintain valid pixel range
            adv_image.data = torch.clamp(
                adv_image.data, (0 - self.mean) / self.std, (1 - self.mean) / self.std)

            adv_image.grad.zero_()

        return adv_image.detach()

    def generate(self, image_path, target_class=None, epsilon=0.01, num_iterations=100):
        """
        Generate an adversarial image targeting a specific class.

        Parameters:
        - image_path: Path to the input image.
        - target_class: Target class name or index. If None, a random class is selected.
        - epsilon: Step size for each iteration (default 0.01).
        - num_iterations: Number of iterations to run the attack (default 100).

        Returns:
        - Tuple of (adversarial PIL Image, added noise tensor, target class name)
        """
        # Load and preprocess the image
        img = Image.open(image_path).convert('RGB')
        img_t = self.preprocess(img)
        input_img = img_t.unsqueeze(0).to(self.device)

        # Classify the original image
        orig_pred_idx, orig_pred_class = self.classify(img)

        # Normalize the image
        input_img_norm = (input_img - self.mean) / self.std

        # Select target class
        if target_class is None:
            target_idx, target_name = self.get_random_class(exclude_class=orig_pred_idx)
            print(f"Randomly selected target class: {target_name} (class index: {target_idx})")
        elif isinstance(target_class, str):
            target_idx = self.get_class_index(target_class)
            target_name = target_class
        elif isinstance(target_class, int):
            target_idx = target_class
            target_name = self.idx2label[target_idx]
        else:
            raise ValueError(
                "target_class must be None, a string (class name), or an integer (class index).")

        # Generate adversarial image
        adv_img_norm = self.targeted_attack(
            input_img_norm, target_idx, epsilon, num_iterations)

        # Calculate the noise
        noise = adv_img_norm - input_img_norm

        # Unnormalize the adversarial image
        adv_img = adv_img_norm * self.std + self.mean
        adv_img = torch.clamp(adv_img, 0, 1)

        # Convert tensor to PIL Image
        adv_img_pil = transforms.ToPILImage()(adv_img.squeeze().cpu())

        return adv_img_pil, noise.cpu(), target_name

    def classify(self, image):
        """
        Classify an image using the model.

        Parameters:
        - image: PIL Image or image tensor.

        Returns:
        - Tuple of (class index, class name).
        """
        if isinstance(image, Image.Image):
            img_t = self.preprocess(image)
            input_img = img_t.unsqueeze(0).to(self.device)
        elif torch.is_tensor(image):
            if image.dim() == 3:
                input_img = image.unsqueeze(0).to(self.device)
            elif image.dim() == 4:
                input_img = image.to(self.device)
            else:
                raise ValueError("Invalid image tensor dimension.")
        else:
            raise ValueError("Invalid image type.")

        # Normalize the image
        input_img_norm = (input_img - self.mean) / self.std

        # Get model prediction
        outputs = self.model(input_img_norm)
        _, preds = outputs.max(1)
        pred_idx = preds.item()
        pred_class = self.idx2label[pred_idx]
        return pred_idx, pred_class

    def visualize_attack(self, image_path, target_class=None, epsilon=0.01, num_iterations=100):
        """
        Generate adversarial image and display the original image, noise, and adversarial image.

        Parameters:
        - image_path: Path to the input image.
        - target_class: Target class name or index. If None, a random class is selected.
        - epsilon: Step size for each iteration.
        - num_iterations: Number of iterations to run the attack.
        """
        # Generate adversarial image and get the noise
        adv_img_pil, noise, target_name = self.generate(image_path, target_class, epsilon, num_iterations)

        # Load original image
        original_img = Image.open(image_path).convert('RGB')
        original_img_t = self.preprocess(original_img)

        # Convert noise to image
        noise_img = noise.squeeze().permute(1, 2, 0).numpy()
        noise_img = (noise_img - noise_img.min()) / (noise_img.max() - noise_img.min())
        noise_img = (noise_img * 255).astype(np.uint8)
        noise_img_pil = Image.fromarray(noise_img)

        # Display images
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        axs[0].imshow(original_img)
        axs[0].set_title('Original Image')
        axs[0].axis('off')

        axs[1].imshow(noise_img_pil)
        axs[1].set_title('Adversarial Noise')
        axs[1].axis('off')

        axs[2].imshow(adv_img_pil)
        axs[2].set_title('Adversarial Image')
        axs[2].axis('off')

        plt.suptitle(f"Target Class: {target_name}", fontsize=16)
        plt.show()