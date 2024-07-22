import os

class SavePaths:
    def __init__(self):
        # Automatically determine the project root based on the location of this file
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))  # Project root directory
        self.base_output_dir = os.path.join(self.project_root, 'output')  # Output directory
        self._initialize_base_dir()

    def _initialize_base_dir(self):
        """Initialize the base output directory and create predefined subdirectories."""
        if not os.path.exists(self.base_output_dir):
            os.makedirs(self.base_output_dir)
        
        # List of predefined subdirectories
        predefined_dirs = ['fits', 'hdf5', 'txt', 'plot']
        for directory in predefined_dirs:
            dir_path = os.path.join(self.base_output_dir, directory)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

    def get_filetype_path(self, file_type):
        """Get the path for the specified file type and create the directory if not present.

        Args:
            file_type (str): Type of file ('fits', 'hdf5', 'txt', 'plot').

        Returns:
            str: Path to the directory for the file type.
        """
        if file_type not in ['fits', 'hdf5', 'txt', 'plot']:
            raise ValueError(f"Invalid file type '{file_type}'. Must be one of 'fits', 'hdf5', 'txt', 'plot'.")
        
        directory = os.path.join(self.base_output_dir, file_type)
        if not os.path.exists(directory):
            os.makedirs(directory)
        return directory

    def create_subdir(self, path, subdirname):
        """Create a subdirectory within the specified path.

        Args:
            path (str): The path where the subdirectory will be created.
            subdirname (str): The name of the subdirectory to create.

        Returns:
            str: Full path to the created subdirectory.
        """
        directory = os.path.join(path, subdirname)
        if not os.path.exists(directory):
            os.makedirs(directory)
        return directory