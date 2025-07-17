# AI Image Description for the Visually Impaired
# This script runs in PyCharm IDE and uses a pre-trained vision-language model to generate image descriptions.

import matplotlib.pyplot as plt
import numpy as np
# Import necessary libraries
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import mean_squared_error
from transformers import BlipProcessor, BlipForConditionalGeneration

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Define the project outline
print("""
### Project Outline
- Problem statement for AI image description
- Loading and exploring an image dataset
- Preprocessing images for the AI model
- Generating image descriptions using a pre-trained model
- Evaluating description quality (optional, qualitative)
- Saving descriptions to a file
""")

# Problem Statement
print("""
## Problem Statement
You are tasked with creating an automated system to generate descriptive captions for images to assist visually impaired users. The system should process images from a dataset, generate accurate and detailed text descriptions, and save them for accessibility purposes. The descriptions must be clear, concise, and informative, capturing key visual elements. You are provided with a dataset of images (e.g., a CSV file with image paths and optional metadata) to test the system.
""")

# Load dataset
# Assume a CSV file with columns: 'image_path' (relative or absolute path to images) and optional 'category' or 'description'
dataset_path = 'image_dataset.csv'  # Update with actual path
try:
    image_df = pd.read_csv(dataset_path)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: Dataset file not found. Creating a sample dataset.")
    # Create a sample dataset for demonstration
    image_df = pd.DataFrame({
        'image_path': ['images/sample1.jpg', 'images/sample2.jpg', 'images/sample3.jpg'],
        'category': ['outdoor', 'indoor', 'portrait']
    })

# Display dataset info
print("\nDataset Info:")
print(image_df.info())
print("\nFirst few rows of the dataset:")
print(image_df.head())

# Display summary statistics for categorical columns (if any)
if 'category' in image_df.columns:
    print("\nCategory Distribution:")
    print(image_df['category'].value_counts())

# Load pre-trained BLIP model and processor
print("\nLoading BLIP model for image captioning...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


# Function to preprocess and generate description for a single image
def generate_image_description(image_path):
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        inputs = processor(images=image, return_tensors="pt").to(device)

        # Generate description
        with torch.no_grad():
            output = model.generate(**inputs, max_length=50)
        description = processor.decode(output[0], skip_special_tokens=True)
        return description
    except Exception as e:
        return f"Error processing {image_path}: {str(e)}"


# Generate descriptions for all images in the dataset
print("\nGenerating image descriptions...")
image_df['description'] = image
deposited_df['image_path'].apply(generate_image_description)

# Display the dataset with generated descriptions
print("\nDataset with Generated Descriptions:")
print(image_df[['image_path', 'description']].head())

# Save descriptions to a new CSV file
output_path = 'image_descriptions.csv'
image_df.to_csv(output_path, index=False)
print(f"\nDescriptions saved to {output_path}")

# Optional: Visualize category distribution (if applicable)
if 'category' in image_df.columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=image_df, x='category')
    plt.title('Distribution of Image Categories')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Optional: Qualitative evaluation
# Since there's no ground truth for descriptions, we can print a few for manual review
print("\nSample Descriptions for Review:")
for idx, row in image_df.head(5).iterrows():
    print(f"Image: {row['image_path']}")
    print(f"Description: {row['description']}\n")

# Optional: If ground truth descriptions exist, compute a simple metric (e.g., string similarity)
# This requires a library like `textdistance` and a 'true_description' column
try:
    # noinspection PyUnresolvedReferences
    import textdistance

    if 'true_description' in image_df.columns:
        def compute_similarity(row):
            return textdistance.levenshtein.normalized_similarity(
                row['description'], row['true_description']
            )


        image_df['similarity'] = image_df.apply(compute_similarity, axis=1)
        avg_similarity = image_df['similarity'].mean()
        print(f"\nAverage Description Similarity: {avg_similarity:.4f}")
except ImportError:
    print("\nTextdistance library not installed. Skipping similarity computation.")

print("\nImage description generation complete!")