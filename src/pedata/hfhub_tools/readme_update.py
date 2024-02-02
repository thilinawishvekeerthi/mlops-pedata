# setting setting the __package__ attribute to solve the relative import proplem when runningn the scripts in the command line
__package__ = "pedata.hfhub_tools"
import os
from huggingface_hub import (
    hf_hub_download,
    upload_file,
    upload_folder,
)
from .constants import README_FORMATTING


class ReadMe:
    """A class to handle the readme file"""

    def __init__(self, local_dir: str = None):
        """Initialize the class

        Args:
            local_dir (str): The local folder containing the readme file.

        Raises:
            ValueError: readme_path must be specified
        """
        if local_dir is None:
            raise ValueError("readme_path must be specified")

        self._local_dir = os.path.abspath(local_dir)
        self._readme_path = os.path.join(local_dir, "README.md")
        self._readme_content = ""

    @property
    def readme_content(self) -> str:
        """The content of the readme"""
        return self._readme_content

    @property
    def readme_path(self) -> str:
        """The path to the readme file"""
        return self._readme_path

    @property
    def figure_path(self) -> str:
        """The path to the figures"""
        return self._figure_path

    @staticmethod
    def insert_section(
        section_start, section_end, updated_section, readme_content
    ) -> None:
        """Insert the updated section in the readme

        Args:
            section_start: The start of the section to update
            section_end: The end of the section to update
            updated_section: The updated section
            readme_content: The content of the readme

        Returns:
            The updated content of the readme
        """
        return (
            readme_content[:section_start]
            + updated_section
            + readme_content[section_end:]
        )

    @staticmethod
    def write_readme(readme_path, updated_readme_content) -> None:
        """Write the updated readme content to the readme file

        Args:
            readme_path: The path to the readme file
            updated_readme_content: The updated content of the readme
        """
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(updated_readme_content)

    def create_readme() -> None:
        # TODO
        pass

    def pull_readme_from_hub(self, repo_id: str = None, repo_type="dataset") -> None:
        """Pulls the readme file from the hub into the local directory

        Args:
            repo_id: The id of the repo to pull the readme from
            repo_type: The type of repo to pull the readme from. Defaults to "dataset"

        Returns:
            None
        """
        if repo_id is None:
            raise ValueError("repo_id must be provided")

        self.repo_id = repo_id
        self.repo_type = repo_type

        _ = hf_hub_download(
            repo_id=repo_id,
            filename="README.md",
            repo_type=self.repo_type,
            local_dir=self._local_dir,
        )
        # return the path to the readme file
        self._readme_path = os.path.join(self._local_dir, "README.md")
        return self.readme_path

    def update_readme(
        self,
        section_elements: list,
        section_name: str = "features",
    ) -> None:
        """Update the section of the readme with the updated information

        Args:
            section_list (list): The updated features section
            section_name (str): The name of the section to update.
        """
        with open(self.readme_path, "r", encoding="utf-8") as f:
            readme_content = f.read()

        # Create the new section
        element_info = "".join(
            [
                f"- {element}{README_FORMATTING['in_section_sep']}"
                for element in section_elements
            ]
        )

        # Find and replace the element section with the updated information
        section_start = readme_content.find(f"{README_FORMATTING[section_name]}")
        section_end = readme_content.find("\n\n\n", section_start)
        updated_section = f"{README_FORMATTING[section_name]}{README_FORMATTING['section_end']}{element_info}{README_FORMATTING['section_end']}"

        self._readme_content = self.insert_section(
            section_start, section_end, updated_section, readme_content
        )
        self.write_readme(self.readme_path, self._readme_content)

    def push_readme_to_hub(
        self,
        repo_id: str = None,
        verbose=True,
    ) -> None:
        """Pushes the readme file to the hub

        Args:
            readme_path: The path to the readme file. If None, defaults to
                the README.md file in the current directory
            repo_id: The id of the repo to push the readme to
                default to None, which sets it to the repo_id used to pull the readme

        Returns:
            None

        """
        if repo_id is None:
            repo_id = self.repo_id

        upload_file(
            path_or_fileobj=self.readme_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type=self.repo_type,
        )
        if verbose:
            print(f"Pushed {self.readme_path} to {repo_id}")

    def update_readme_figures(
        self,
        figure_path: str = None,
    ) -> None:
        """Update the figures section of the readme with the updated information

        Args:
            figure_path: The directory containing the figures to add to the readme

        Returns:
            None

        Raises:
            ValueError: figure_path must be specified
        """

        def make_relative_path(base_path, target_path):
            return os.path.relpath(target_path, os.path.dirname(base_path))

        if figure_path is None:
            raise ValueError("readme_path must be specified")

        self._figure_path = os.path.abspath(figure_path)
        self.figure_rel_path = make_relative_path(self.readme_path, figure_path)

        with open(self.readme_path, "r", encoding="utf-8") as f:
            readme_content = f.read()

        # Create the new section
        figure_names = os.listdir(figure_path)
        figure_info = "".join(
            [
                f'<img src="{self.figure_rel_path}/{figure_name}"> {README_FORMATTING["in_section_sep"]}'
                for figure_name in figure_names
            ]
        )

        # Find and replace the element section with the updated information
        section_start = readme_content.find(f"{README_FORMATTING['figures']}")
        section_end = readme_content.find("\n\n\n", section_start)

        updated_section = f"{README_FORMATTING['section_end']}{README_FORMATTING['figures']}{README_FORMATTING['in_section_sep']}{figure_info}{ README_FORMATTING['section_end']}"

        self._readme_content = self.insert_section(
            section_start, section_end, updated_section, readme_content
        )
        self.write_readme(self.readme_path, self._readme_content)

    def push_figures_to_hub(self) -> None:
        """Pushes the figures to the hub"""
        upload_folder(
            folder_path=self._figure_path,
            path_in_repo=self.figure_rel_path,
            repo_id=self.repo_id,
            repo_type=self.repo_type,
        )


if __name__ == "__main__":  # FIXME: write command line tool for this
    pass
    # # parse arguments
    # parser = argparse.ArgumentParser(
    #     description="Create and push a dataset to Hugging Face."
    # )
    # # required arguments
    # parser.add_argument(
    #     "--repo",
    #     required=True,
    #     help="Name of the repository to pull from on Hugging Face.",
    # )

    # parser.add_argument(
    #     "--update_just_readme",
    #     required=False,
    #     help="When update - just update the readme or dataset card.",
    #     default=False,
    # )

    # parser.add_argument(
    #     "--splits_to_combine_as_whole_ds",
    #     required=False,
    #     help="list: names of the split to combine as 'whole_dataset'",
    #     nargs="+",
    #     default=[],
    # )

    # args = parser.parse_args()
    # print(args)
    # # create dataset upload object
    # data_upload = DatasetUpload()
