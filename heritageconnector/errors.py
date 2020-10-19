def raise_must_be_str_or_list(var_name: str):
    raise ValueError(f"Variable {var_name} must be either string or list.")


def raise_file_not_found_error(file_path: str, file_or_folder="folder"):
    raise FileNotFoundError(f"{file_or_folder.title()} {file_path} does not exist.")
