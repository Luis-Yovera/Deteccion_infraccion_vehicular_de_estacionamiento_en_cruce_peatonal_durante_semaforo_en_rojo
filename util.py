import csv

def write_csv(results, output_path):
    """
    Escribe los resultados en un archivo CSV.

    Args:
        results (dict): Diccionario que contiene los resultados.
        output_path (str): Ruta del archivo CSV de salida.
    """
    # Abrir el archivo CSV en modo escritura con codificación UTF-8
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Escribir la cabecera del archivo CSV
        writer.writerow(['frame_nmr', 'license_box', 'license_plate', 'license_confidence'])

        # Recorrer cada entrada en el diccionario de resultados
        for frame_nmr, frame_data in results.items():
            # Verificar si existe información de 'license' y 'plate' en el frame actual
            if 'license' in frame_data and \
               'plate' in frame_data['license'] and \
               frame_data['license']['plate'] != "No text":

                # Extraer datos necesarios
                box = frame_data['license']['box']
                plate = frame_data['license']['plate']
                confidence = frame_data['license']['confidence']

                # Escribir una fila con los datos
                writer.writerow([
                    frame_nmr,
                    '[{} {} {} {}]'.format(box[0], box[1], box[2], box[3]),
                    plate,
                    confidence
                ])
