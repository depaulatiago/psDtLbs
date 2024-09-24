import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import os
from PIL import Image
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import shutil

# Definir transformações de pré-processamento
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Função para processar nova imagem e gerar embedding
def process_new_image(img_path):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tensor = transform(img_pil)
    img_tensor = img_tensor.unsqueeze(0)
    with torch.no_grad():
        embedding = model(img_tensor)
    return embedding.squeeze().numpy()

# Função para gerar embeddings
def generate_embeddings(dataloader):
    embeddings = []
    labels = []
    image_paths = []
    with torch.no_grad():
        for imgs, label in dataloader:
            outputs = model(imgs)
            embeddings.append(outputs.numpy())
            labels.extend(label.numpy())
            for img_path in dataloader.dataset.samples:
                image_paths.append(img_path[0])
    return np.vstack(embeddings), labels, image_paths

# Função para adicionar uma nova pessoa ao banco de dados
def add_new_person(img_path, new_label):
    new_embedding = process_new_image(img_path)
    existing_embeddings = np.load('celebrity_embeddings.npy') if os.path.exists('celebrity_embeddings.npy') else np.empty((0, 512))
    existing_labels = np.load('celebrity_labels.npy') if os.path.exists('celebrity_labels.npy') else np.empty(0, dtype=int)
    updated_embeddings = np.vstack([existing_embeddings, new_embedding])
    updated_labels = np.append(existing_labels, new_label)
    np.save('celebrity_embeddings.npy', updated_embeddings)
    np.save('celebrity_labels.npy', updated_labels)

# Função para reconhecer uma nova imagem
def recognize_face(new_embedding, database, labels):
    similarities = [np.dot(new_embedding, db_embedding) for db_embedding in database]
    idx = np.argmax(similarities)
    return idx, similarities[idx]

# Função para mostrar imagens com o score da métrica
def show_images(img1_path, img2_path, score):
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Imagem com Máscara")
    plt.imshow(img1)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"Imagem Identificada\nScore: {score:.4f}")
    plt.imshow(img2)
    plt.axis('off')

    plt.show()

# Função para verificar arquivos válidos (ignorar .ipynb_checkpoints e outros inválidos)
def is_valid_file(file_path):
    valid_extensions = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp']
    # Ignorar arquivos em qualquer diretório chamado '.ipynb_checkpoints'
    if '.ipynb_checkpoints' in file_path:
        return False
    return any(file_path.endswith(ext) for ext in valid_extensions)


# Definir o caminho do dataset e criar DataLoader
dataset_path = "/content/drive/MyDrive/pessoasA/"
dataset = datasets.ImageFolder(root=dataset_path, transform=transform, is_valid_file=is_valid_file)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Definir o modelo
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
num_classes = len(dataset.classes)
model.fc = nn.Linear(num_features, num_classes)  # Ajustar a camada final para o número de classes

# Carregar o modelo salvo, se existir
if os.path.exists('celebrity_model.pth'):
    # Certifique-se de que o número de classes corresponde ao modelo salvo
    saved_model = torch.load('celebrity_model.pth', weights_only=True)
    if 'fc.weight' in saved_model:
        saved_num_classes = saved_model['fc.weight'].size(0)
        if saved_num_classes != num_classes:
            raise ValueError(f"O número de classes no modelo salvo ({saved_num_classes}) é diferente do número de classes esperado ({num_classes}).")
    model.load_state_dict(saved_model)
    print("Modelo pré-treinado carregado.")
else:
    print("Treinando novo modelo...")

# Definir o critério e o otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Treinar o modelo (se necessário)
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

# Salvar o modelo treinado (atualizando o arquivo anterior)
torch.save(model.state_dict(), 'celebrity_model.pth')

# Gerar embeddings para o banco de dados
embeddings, labels, image_paths = generate_embeddings(dataloader)
np.save('celebrity_embeddings.npy', embeddings)
np.save('celebrity_labels.npy', labels)

# Adicionar uma nova pessoa ao banco de dados (substitua o caminho e o label)
new_person_image_path = '/content/drive/MyDrive/mascara/novapessoa.jpg'
new_person_label = 5
add_new_person(new_person_image_path, new_person_label)

# Reconhecer uma nova imagem (com máscara)
masked_image_path = '/content/drive/MyDrive/mascara/marcelinhomascara.jpg'
new_embedding = process_new_image(masked_image_path)

# Carregar embeddings e labels do banco de dados
database_embeddings = np.load('celebrity_embeddings.npy')
database_labels = np.load('celebrity_labels.npy')

# Identificar a pessoa
identity, score = recognize_face(new_embedding, database_embeddings, database_labels)

# Mostrar a imagem com máscara e a imagem identificada com o score da métrica
identified_image_path = image_paths[int(identity)]
show_images(masked_image_path, identified_image_path, score)
