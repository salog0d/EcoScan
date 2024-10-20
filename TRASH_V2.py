import os
import torch
import torchvision
from torch.utils.data import random_split
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 1. Especifica el directorio donde se encuentran tus imágenes
data_dir = 'C:/Oscar-Desktop/Proyectos/Hackaton 2024/Images/Recyclable and Household Waste Classification'

# 2. Lista las clases de imágenes
classes = os.listdir(data_dir)
print(classes)

# 3. Transformaciones para el preprocesamiento de las imágenes
transformations = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# 4. Cargar el dataset con tus imágenes
dataset = ImageFolder(data_dir, transform=transformations)

# 5. Mostrar una muestra del dataset
def show_sample(img, label):
    print("Label:", dataset.classes[label], "(Class No: " + str(label) + ")")
    plt.imshow(img.permute(1, 2, 0))

img, label = dataset[12]
show_sample(img, label)

# 6. Dividir el dataset en entrenamiento, validación y prueba
random_seed = 42
torch.manual_seed(random_seed)
train_ds, val_ds, test_ds = random_split(dataset, [10799, 1349, 1351])
print(len(dataset), len(train_ds), len(val_ds), len(test_ds))

# 7. Crear DataLoaders para cargar los datos en lotes
from torch.utils.data.dataloader import DataLoader
batch_size = 32

train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)

# 8. Mostrar un lote de imágenes
from torchvision.utils import make_grid

def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
        break

# 9. Definir la métrica de precisión
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

# 10. Definir una clase base para la clasificación de imágenes
class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generar predicciones
        loss = F.cross_entropy(out, labels) # Calcular pérdida
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generar predicciones
        loss = F.cross_entropy(out, labels)   # Calcular pérdida
        acc = accuracy(out, labels)           # Calcular precisión
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combinar pérdidas
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combinar precisiones
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch {}: train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch+1, result['train_loss'], result['val_loss'], result['val_acc']))

# 11. Definir la arquitectura de ResNet
class ResNet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = models.resnet50(pretrained=True)
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, len(dataset.classes))  # Reemplazar la última capa
    
    def forward(self, xb):
        return torch.sigmoid(self.network(xb))

# 12. Enviar los datos y el modelo al dispositivo adecuado (GPU o CPU)
def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)

device = get_default_device()
train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)
model = to_device(ResNet(), device)

# 13. Función para evaluar el modelo
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

# 14. Entrenamiento del modelo
def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}...")
        model.train()
        train_losses = []

        for batch_idx, batch in enumerate(train_loader):  # Start of the batch loop
            images, labels = batch
            loss = model.training_step(batch)
            train_losses.append(loss)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Print current batch and loss
            print(f"  Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}")

        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    
    return history

# 15. Entrenar y evaluar el modelo
if __name__ == '__main__':
    num_epochs = 8
    opt_func = torch.optim.Adam
    lr = 5.5e-5
    history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)

    # 16. Graficar las precisiones
    def plot_accuracies(history):
        accuracies = [x['val_acc'] for x in history]
        plt.plot(accuracies, '-x')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title('Accuracy vs. No. of epochs')

    plot_accuracies(history)

    # 17. Graficar las pérdidas
    def plot_losses(history):
        train_losses = [x.get('train_loss') for x in history]
        val_losses = [x['val_loss'] for x in history]
        plt.plot(train_losses, '-bx')
        plt.plot(val_losses, '-rx')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['Training', 'Validation'])
        plt.title('Loss vs. No. of epochs')

    plot_losses(history)

    # 18. Predicción de imágenes individuales
    def predict_image(img, model):
        xb = to_device(img.unsqueeze(0), device)
        yb = model(xb)
        prob, preds = torch.max(yb, dim=1)
        return dataset.classes[preds[0].item()]

    from PIL import Image

    # 1. Cargar la imagen que deseas verificar
    img_path = 'C:/Users/oscar/Downloads/3.jpg'  # Cambia a la ruta de tu imagen
    img = Image.open(img_path)

    # 2. Aplicar las mismas transformaciones que usaste para entrenar el modelo
    img_tensor = transformations(img)

    # 3. Hacer la predicción
    label = predict_image(img_tensor, model)
    print(f'Predicted label: {label}')