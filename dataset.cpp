#include "dataset.h"
#include <fstream>
#include <set>

// Function to read text from a file
std::string read_txt(const std::string &path) {
    std::ifstream file(path);

    if (!file.is_open()) {
        std::cerr << "Failed to open the file: " << path << std::endl;
        return "";
    }
    std::string text = "";
    std::string line;

    while (std::getline(file, line)) {
        text += line + "\n";
    }

    file.close();
    return text;
}

// Constructor
Dataset::Dataset(const std::string &txt_path, size_t block_size) {
    raw_data = read_txt(txt_path);
    this->block_size = block_size;
    std::set<char> char_set(raw_data.begin(), raw_data.end());
    vocab = std::string(char_set.begin(), char_set.end());
    auto encoded = encode(raw_data);
    size_t n = static_cast<size_t>(raw_data.length() * 0.9);

    train_data = std::vector<size_t>(encoded.begin(), encoded.begin() + n);
    val_data = std::vector<size_t>(encoded.begin() + n, encoded.end());
}

// Encode function
std::vector<size_t> Dataset::encode(const std::string &txt) const {
    std::vector<size_t> encoded;
    for (char c : txt) {
        size_t idx = vocab.find(c);
        if (idx != std::string::npos) {
            encoded.push_back(idx);
        }
        else {
            // Handle unknown characters, e.g., skip or assign a special token
            // Here, we'll skip
            continue;
        }
    }
    return encoded;
}

// Decode function
std::string Dataset::decode(const std::vector<size_t> &idxs) const {
    std::string decoded;
    for (size_t idx : idxs) {
        if (idx < vocab.size()) {
            decoded += vocab[idx];
        }
        else {
            // Handle invalid indices, e.g., append a placeholder
            decoded += '?';
        }
    }
    return decoded;
}

// Get training data
std::vector<size_t> Dataset::get_train() const {
    return train_data;
}

// Get validation data
std::vector<size_t> Dataset::get_val() const {
    return val_data;
}

// Get vocabulary size
size_t Dataset::get_vocab_size() const {
    return vocab.size();
}
