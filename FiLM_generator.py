import torch
import torch.nn as nn


def mlp(num_inputs,
        num_outputs,
        hidden_layers,
        activation=nn.ReLU,
        output_activation=nn.Identity):
    layers = []
    layer_sizes = [num_inputs] + hidden_layers + [num_outputs]
    for i in range(len(layer_sizes) - 1):
        act = activation if i < len(layer_sizes) - 2 else output_activation
        layers += [nn.Linear(layer_sizes[i], layer_sizes[i + 1]), act()]
    return nn.Sequential(*layers)


class FiLMGenerator(nn.Module):
    num_beta_gamma = 2  # one scale and one bias

    def __init__(self, film_input_dim, num_params_to_film, hidden_layers):
        super().__init__()
        self.film_input_dim = film_input_dim
        self.num_params_to_film = num_params_to_film
        self.film_output_dim = self.num_beta_gamma * num_params_to_film

        self.mlp = mlp(num_inputs=self.film_input_dim,
                       num_outputs=self.film_output_dim,
                       hidden_layers=hidden_layers)

    def forward(self, x):
        assert(len(x.shape) == 2)
        assert(x.shape[1] == self.film_input_dim)
        batch_size = x.shape[0]

        # Use delta-gamma so baseline is gamma=1
        film_output = self.mlp(x)
        beta, delta_gamma = torch.chunk(film_output, chunks=self.num_beta_gamma, dim=1)
        gamma = delta_gamma + 1.
        assert(beta.shape == gamma.shape == (batch_size, self.num_params_to_film))

        return beta, gamma


if __name__ == "__main__":
    import FiLM_resnet
    from torchinfo import summary

    BATCH_SIZE = 1
    FILM_INPUT_DIM = 100
    NUM_BETA_GAMMA = 2  # 1 for beta, 1 for gamma
    IMG_SHAPE = (3, 224, 224)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # FiLM-ed resnet
    filmed_resnet = FiLM_resnet.resnet18(weights=FiLM_resnet.ResNet18_Weights.DEFAULT).to(device)
    print(f"num_film_params = {filmed_resnet.num_film_params}")

    # FiLM generator
    film_generator = FiLMGenerator(film_input_dim=FILM_INPUT_DIM,
                                   num_params_to_film=filmed_resnet.num_film_params,
                                   hidden_layers=[256, 256]).to(device)

    print("~" * 100)
    print(f"Summary of FiLMed resnet:")
    print("~" * 100)
    summary(filmed_resnet, input_size=(BATCH_SIZE, *IMG_SHAPE), depth=float('inf'), device=device)

    print("~" * 100)
    print(f"Summary of FiLM generator:")
    print("~" * 100)
    summary(film_generator, input_size=(BATCH_SIZE, FILM_INPUT_DIM), depth=float('inf'), device=device)

    # Generate beta and gamma
    example_film_input = torch.randn(BATCH_SIZE, FILM_INPUT_DIM).to(device)
    beta, gamma = film_generator(example_film_input)
    print(f"beta.shape = {beta.shape}")
    print(f"gamma.shape = {gamma.shape}")

    example_img_input = torch.randn(BATCH_SIZE, *IMG_SHAPE).to(device)
    example_img_output = filmed_resnet(example_img_input, beta=beta, gamma=gamma)

    print(f"Output shape = {example_img_output.shape}")