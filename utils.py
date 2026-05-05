import os

def remove_folder(folder_path):
    """
    Deletes a folder and all its contents.
    Returns:
        bool: True if folder was deleted, False otherwise.
    """
    if not folder_path or not os.path.exists(folder_path):
        return False

    # Delete all files
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for f in files:
            os.remove(os.path.join(root, f))
        for d in dirs:
            os.rmdir(os.path.join(root, d))
    os.rmdir(folder_path)
    print(f"Deleted folder: {folder_path}")
    return True

def translate_classification(classification):
    lower = classification.lower()
    if lower == "snowshoe hare":
        return "Rabbit"
    elif lower == "domestic cat":
        return "Cat"
    elif lower == "domestic dog":
        return "Dog"
    elif lower == "domestic cow":
        return "Cow"
    elif lower == "gray squirrel" or lower == "red squirrel":
        return "Squirrel"
    elif lower == "bird sp.":
        return "Bird"
    elif lower == "mouse sp.":
        return "Mouse"