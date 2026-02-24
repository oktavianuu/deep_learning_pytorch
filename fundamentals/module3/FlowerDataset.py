class FlowerDataset(Dataset):
    """
    A custom dataset class for loading flower image data.

    This class is designed to work with PyTorch's Dataset and DataLoader
    abstractions. It handles loading images and their corresponding labels
    from a specific directory structure.
    """

    def __init__(self, root_dir, transform=None):
        """
        Initializes the dataset object.

        Args:
            root_dir (str): The root directory where the dataset is stored.
            transform (callable, optional): Optional transform to be applied
            on a sample.
        """
        # store the root dir path
        self.root_dir = root_dir
        # Store optional transformation
        self.transform = transform
        # Construct the full path to the image directory.
        self.image_dir = os.path.join(self.root_dir, "jpg")
        # Load and process the labels from the corresponding file.
        self.labels = self.load_and_correct_labels()

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        # The total number of samples is the number of labels.
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset at the specified index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the image and its label.
        """
        # Retrieve the image for the given index.
        image = self.retrieve_image(idx)

        # check transform
        if self.transform is not None:
            # Apply the transform to the image.
            image = self.transform(image)

        # Get the label corresponding to the index.
        label = self.labels[idx]

        # return the processed image and its label 
        return image, label
    
    def retrieve_image(self, idx):
        """
        Loads a single image from disk based on its index.

        Args:
            idx (int): The index of the image to load.

        Returns:
            PIL.Image.Image: The loaded image, converted to RGB.
        """
        img_name = f"image_{idx + 1:05d}.jpg"

        # Construct the full path to the image file.
        img_path = os.path.join(self.image_dir, img_name)

        # Open the file image
        with Image.open(img_path) as img:
            # Convert the image to the RGB color space and return it.
            image = img.convert("RGB")
        return image
    

    def load_and_correct_labels(self):
        """
        Loads labels from a .mat file and adjusts them to be zero-indexed.

        Returns:
            numpy.ndarray: An array of zero-indexed integer labels.
        """

        # Load the MATLAB file containing the labels 
        self.labels_mat = scipy.io.loadmat(
            os.path.join(self.root_dir, "imagelabels.mat")
        )
        # Extract labels array and correct zero indexing
        labels = self.labels_mat["labels"][0] -1 
        
        #Return the processed labels
        return labels
    
    def get_label_description(self, label):
        """
        Retrieves the text description for a given label index.

        Args:
            label (int): The integer label.

        Returns:
            str: The corresponding text description of the label.
        """
        # Construct the path to the file containing label descriptions.
        path_labels_description = os.path.join(self.root_dir, "labels_description.txt")
        # Open the label desc file for reading
        with open(path_labels_description, "r") as f:
            # Read all lines from the file
            lines = f.readlines()
        
        # Get the description for the specified label and remove leading/trailing whitespace.
        description = lines[label].strip()
        # Return the clean desc 
        return description