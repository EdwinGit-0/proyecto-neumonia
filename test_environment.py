import sys

REQUIRED_PYTHON = "python3"


def main():
    system_major = sys.version_info.major
    if REQUIRED_PYTHON == "python":
        required_major = 2
    elif REQUIRED_PYTHON == "python3":
        required_major = 3
    else:
        raise ValueError("Intérprete de Python no reconocido: {}".format(
            REQUIRED_PYTHON))

    if system_major != required_major:
        raise TypeError(
            "Este proyecto requiere Python {}. Se encontró: Python {}".format(
                required_major, sys.version))
    else:
        print(">>> El entorno de desarrollo pasa todas las pruebas!")


if __name__ == '__main__':
    main()
