import os
import numpy as np
from scipy.io import savemat


def convert_npy_to_mat_recursive(input_folder, output_folder):
    """
    Convierte todos los archivos .npy en una estructura de carpetas anidada a .mat,
    preservando la estructura de carpetas en la carpeta de salida.

    Args:
        input_folder (str): Carpeta raíz de los archivos .npy.
        output_folder (str): Carpeta raíz para guardar los archivos .mat.
    """
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.npy'):
                # Ruta completa del archivo .npy
                npy_path = os.path.join(root, file)

                # Crear la estructura de carpetas equivalente en la carpeta de salida
                relative_path = os.path.relpath(root, input_folder)  # Subcarpetas relativas al input_folder
                output_subfolder = os.path.join(output_folder, relative_path)
                os.makedirs(output_subfolder, exist_ok=True)

                # Ruta de salida para el archivo .mat
                mat_path = os.path.join(output_subfolder, file.replace('.npy', '.mat'))

                # Cargar y convertir el archivo
                data = np.load(npy_path)
                savemat(mat_path, {'data': data})

                print(f"Convertido: {npy_path} -> {mat_path}")


# Ejemplo de uso
input_folder = "data"  # Carpeta raíz con carpetas y archivos .npy
output_folder = "data_mat"  # Carpeta raíz para guardar los archivos .mat
convert_npy_to_mat_recursive(input_folder, output_folder)