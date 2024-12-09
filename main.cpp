#include "bigram_lm.h"
#include "dataset.h"
#include <random>
#include <string>
#include <iostream>
#include <fstream>

std::vector<size_t> random_int(size_t min, size_t max, size_t size) {
  std::vector<size_t> random_numbers;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> dist(min, max);
  for (int i = 0; i < size; i++) {
    random_numbers.emplace_back(dist(gen));
  }
  return random_numbers;
}

std::pair<torch::Tensor, torch::Tensor> get_batch(std::vector<size_t> &split,
                                                  const int &block_size,
                                                  const int &batch_size) {
  auto data_ptr = split.data();
  std::vector<torch::Tensor> xb;
  std::vector<torch::Tensor> yb;
  auto random_idxs = random_int(0, split.size() - block_size, batch_size);
  for (auto &i : random_idxs) {
    xb.emplace_back(torch::from_blob(data_ptr + i, {block_size}, torch::kLong));
    yb.emplace_back(
        torch::from_blob(data_ptr + (i + 1), {block_size}, torch::kLong));
  }
  return std::make_pair(torch::stack(xb), torch::stack(yb));
}

// Function to generate text
std::string generate_text(
    std::shared_ptr<BigramLM> model,
    const std::string& seed,
    size_t generate_length,
    const Dataset& dataset
) {
    std::string generated = seed;
    std::vector<size_t> input_ids = dataset.encode(seed);

    // Move model to evaluation mode
    model->eval();

    // Disable gradient computation for inference
    torch::NoGradGuard no_grad;

    for (size_t i = 0; i < generate_length; ++i) {
        // Convert input_ids to tensor
        torch::Tensor input_tensor = torch::from_blob(
            input_ids.data(),
            {1, static_cast<long>(input_ids.size())},
            torch::kLong
        ).clone(); // Clone to ensure memory safety

        // Forward pass
        torch::Tensor logits = model->forward(input_tensor);

        // Get the last timestep's logits
        torch::Tensor last_logits = logits.slice(/*dim=*/1, /*start=*/-1, /*end=*/-1).squeeze(1); // Shape: [1, vocab_size]

        // Apply softmax to get probabilities
        torch::Tensor probs = torch::softmax(last_logits, /*dim=*/1);

        // Sample from the distribution
        torch::Tensor sampled_id = torch::multinomial(probs, /*num_samples=*/1);

        // Get the sampled token
        size_t token = sampled_id.item<long>();

        // Append to the generated text
        generated += dataset.decode({token});

        // Append to input_ids for the next prediction
        input_ids.push_back(token);
    }

    return generated;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <mode> [options]" << std::endl;
        std::cerr << "Modes:" << std::endl;
        std::cerr << "  train     Train the model" << std::endl;
        std::cerr << "  generate  Generate text using the trained model" << std::endl;
        return 1;
    }

    std::string mode = argv[1];

    if (mode == "train") {
        // Training mode
        std::string dataset_path = "../dataset/custom_dataset/input.txt"; // Ensure this path is correct
        int block_size = 8;
        int batch_size = 4;
        size_t num_iterations = 100000;

        Dataset dataset(dataset_path, block_size);
        auto train_data = dataset.get_train();
        auto val_data = dataset.get_val();

        auto model = std::make_shared<BigramLM>(dataset.get_vocab_size());

        torch::optim::AdamW optimizer(model->parameters(), 0.001);

        // Optionally load from checkpoint
        if (std::ifstream("checkpoint.pt")) {
            try {
                torch::load(model, "checkpoint.pt");
                torch::load(optimizer, "optimizer.pt");
                std::cout << "Checkpoint loaded successfully. Resuming training." << std::endl;
            } catch (const c10::Error& e) {
                std::cerr << "Error loading the checkpoint: " << e.what() << std::endl;
            }
        } else {
            std::cout << "No checkpoint found. Starting fresh training." << std::endl;
        }

        // Open a log file
        std::ofstream loss_log("loss_log.csv");
        loss_log << "Iteration,Loss\n";

        for (size_t i = 1; i <= num_iterations; ++i) {
            optimizer.zero_grad();
            auto [xb, yb] = get_batch(train_data, block_size, batch_size);
            torch::Tensor loss = model->forward(xb, yb);
            loss.backward();
            optimizer.step();

            if (i % 1000 == 0) {
                std::cout << "Iter: " << i << " | Loss: " << loss.item<float>() << std::endl;
                loss_log << i << "," << loss.item<float>() << "\n";
                torch::save(model, "checkpoint.pt");
                torch::save(optimizer, "optimizer.pt");
            }
        }

        loss_log.close();
        std::cout << "Training completed." << std::endl;
    }
    else if (mode == "generate") {
        // Generation mode
        if (argc < 3) {
            std::cerr << "Usage: " << argv[0] << " generate <seed_text> [generate_length]" << std::endl;
            return 1;
        }

        std::string seed = argv[2];
        size_t generate_length = 100; // Default length
        if (argc >= 4) {
            generate_length = std::stoul(argv[3]);
        }

        // Path to the dataset for encoding/decoding
        std::string dataset_path = "/Users/chiragchivate/Documents/QUANT/projects/tiny_jim/nanogpt_cpp-main/tiny_jim.txt";
        int block_size = 8; // Not used in generation but required by Dataset constructor

        Dataset dataset(dataset_path, block_size);

        // Initialize the model
        auto model = std::make_shared<BigramLM>(dataset.get_vocab_size());

        // Load the trained model
        try {
            torch::load(model, "checkpoint.pt");
            torch::load(model, "checkpoint.pt"); // Load entire model
            std::cout << "Model loaded successfully." << std::endl;
        } catch (const c10::Error& e) {
            std::cerr << "Error loading the model: " << e.what() << std::endl;
            return -1;
        }

        // Generate text
        std::string generated_text = generate_text(model, seed, generate_length, dataset);
        std::cout << "Generated Text:\n" << generated_text << std::endl;
    }
    else {
        std::cerr << "Unknown mode: " << mode << std::endl;
        std::cerr << "Available modes: train, generate" << std::endl;
        return 1;
    }

    return 0;
}