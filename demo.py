from FiLM_image_encoder.FiLM_resnet import resnet18, ResNet18_Weights
from FiLM_image_encoder.FiLM_generator import FiLMGenerator
from torchinfo import summary
import torch


def main():
    BATCH_SIZE = 1
    FILM_INPUT_DIM = 100
    IMG_SHAPE = (3, 224, 224)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # FiLM-ed resnet
    filmed_resnet = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
    print(f"num_film_params = {filmed_resnet.num_film_params}")

    # FiLM generator
    film_generator = FiLMGenerator(
        film_input_dim=FILM_INPUT_DIM,
        num_params_to_film=filmed_resnet.num_film_params,
        hidden_layers=[256, 256],
    ).to(device)

    print("~" * 100)
    print(f"Summary of FiLMed resnet:")
    print("~" * 100)
    summary(
        filmed_resnet,
        input_size=(BATCH_SIZE, *IMG_SHAPE),
        depth=float("inf"),
        device=device,
    )

    print("~" * 100)
    print(f"Summary of FiLM generator:")
    print("~" * 100)
    summary(
        film_generator,
        input_size=(BATCH_SIZE, FILM_INPUT_DIM),
        depth=float("inf"),
        device=device,
    )

    # Generate beta and gamma
    example_film_input = torch.randn(BATCH_SIZE, FILM_INPUT_DIM).to(device)
    beta, gamma = film_generator(example_film_input)
    print(f"beta.shape = {beta.shape}")
    print(f"gamma.shape = {gamma.shape}")

    example_img_input = torch.randn(BATCH_SIZE, *IMG_SHAPE).to(device)
    example_img_output = filmed_resnet(example_img_input, beta=beta, gamma=gamma)

    print(f"Output shape = {example_img_output.shape}")


if __name__ == "__main__":
    main()
