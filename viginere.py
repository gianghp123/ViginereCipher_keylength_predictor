import random
import string

def get_random_keyword(min=3, max=25):
    """Generates a random key (3-25 characters default)"""
    key_length = random.randint(min, max)
    keyword = "".join(random.choices(string.ascii_uppercase, k=key_length))
    return keyword.upper()


def _generate_key(msg, key):
    """Generates a key of the same length as the message by repeating the key.

    Args:
        msg (str): The message to encrypt.
        key (str): The key to use for encryption.

    Returns:
        str: The generated key of the same length as the message.
    """
    key = list(key)
    if len(msg) == len(key):
        return key
    else:
        for i in range(len(msg) - len(key)):
            key.append(key[i % len(key)])
    return "".join(key)


def encrypt_vigenere(msg, key):
    """Encrypts a message using the Vigenere cipher.

    This function takes a message and a keyword, and encrypts the message
    using the Vigenere cipher. The message can contain any characters, but
    only letters will be encrypted. Non-letter characters will be left
    unchanged.

    The encryption works by repeating the keyword to match the length of
    the message, and then adding the corresponding characters of the keyword
    to the characters of the message. The result is a new string where each
    letter has been "shifted" by the corresponding letter of the keyword.

    Args:
        msg (str): The message to encrypt.
        key (str): The keyword to use for encryption.

    Returns:
        str: The encrypted message.
    """
    encrypted_text = []
    key = _generate_key(msg, key)
    for i in range(len(msg)):
        char = msg[i]
        if char.isupper():
            encrypted_char = chr(
                (ord(char) + ord(key[i]) - 2 * ord("A")) % 26 + ord("A")
            )
        elif char.islower():
            encrypted_char = chr(
                (ord(char) + ord(key[i]) - 2 * ord("a")) % 26 + ord("a")
            )
        else:
            encrypted_char = char
        encrypted_text.append(encrypted_char)
    return "".join(encrypted_text)


def decrypt_vigenere(msg, key):
    """Decrypts a message using the Vigenere cipher.

    This function takes an encrypted message and a keyword, and decrypts the
    message using the Vigenere cipher. The message can contain any characters,
    but only letters will be decrypted. Non-letter characters will be left
    unchanged.

    The decryption works by repeating the keyword to match the length of the
    message, and then subtracting the corresponding characters of the keyword
    from the characters of the message. The result is a new string where each
    letter has been "unshifted" by the corresponding letter of the keyword.

    Args:
        msg (str): The message to decrypt.
        key (str): The keyword to use for decryption.

    Returns:
        str: The decrypted message.
    """
    decrypted_text = []
    key = _generate_key(msg, key)
    for i in range(len(msg)):
        char = msg[i]
        if char.isupper():
            decrypted_char = chr((ord(char) - ord(key[i]) + 26) % 26 + ord("A"))
        elif char.islower():
            decrypted_char = chr((ord(char) - ord(key[i]) + 26) % 26 + ord("a"))
        else:
            decrypted_char = char
        decrypted_text.append(decrypted_char)
    return "".join(decrypted_text)