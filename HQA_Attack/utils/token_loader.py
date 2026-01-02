def read_hf_token(path: str = "token.txt") -> str:
    """
    Read the Hugging Face API token from a plain-text file.
    """
    try:
        with open(path, encoding="utf-8") as f:
            token = f.read().strip()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Hugging Face token file not found: {path}") from exc

    if not token:
        raise ValueError("Hugging Face token file is empty.")

    return token