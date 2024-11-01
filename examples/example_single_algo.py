"""
Demo: Running an algorithm.
"""
import imaging_server_kit as serverkit

def main():
    client = serverkit.Client()
    status_code = client.connect("http://localhost:8000")
    print(f"{status_code=}")
    print(f"{client.algorithms=}")

    # Get the algo params
    rembg_params = client.get_algorithm_parameters()
    print(f"{rembg_params=}")

    # Get the algo sample image
    rembg_sample_images = client.get_sample_images()
    for image in rembg_sample_images:
        print(f"{image.shape=}")
    sample_image = rembg_sample_images[0]

    # Run the algo (return type is a `LayerDataTuple`)
    data_tuple = client.run_algorithm(
        image=sample_image,
        rembg_model_name="silueta"
    )
    for (data, data_params, data_type) in data_tuple:
        print(f"Algo returned: {data_type=} ({data.shape=})")

if __name__ == "__main__":
    main()
