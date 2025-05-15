#include <tokenizer_lib/tokenizer.h>
#include <algorithm>
#include <set>
#include <queue>
#include <sstream>

Tokenizer::Tokenizer() {
    // Initialize special tokens
    token_to_idx[PAD_TOKEN] = 0;
    token_to_idx[UNK_TOKEN] = 1;
    token_to_idx[BOS_TOKEN] = 2;
    token_to_idx[EOS_TOKEN] = 3;
    
    idx_to_token[0] = PAD_TOKEN;
    idx_to_token[1] = UNK_TOKEN;
    idx_to_token[2] = BOS_TOKEN;
    idx_to_token[3] = EOS_TOKEN;
    
    pad_idx = 0;
    unk_idx = 1;
    bos_idx = 2;
    eos_idx = 3;
}

void Tokenizer::fit_character_level(const std::vector<std::string>& texts) {
    // Collect all unique characters from the corpus
    std::set<char> unique_chars;
    for (const auto& text : texts) {
        unique_chars.insert(text.begin(), text.end());
    }
    
    // Create mappings for each character
    int idx = 4;  // Start from 4 since 0-3 are reserved for special tokens
    for (char c : unique_chars) {
        std::string char_str(1, c);
        token_to_idx[char_str] = idx;
        idx_to_token[idx] = char_str;
        idx++;
    }
}

void Tokenizer::fit_bpe(const std::vector<std::string>& texts, int vocab_size) {
    // First, create character-level vocabulary
    fit_character_level(texts);
    
    // Learn BPE merges
    learn_bpe_merges(texts, vocab_size);
}

std::vector<int> Tokenizer::encode_character_level(const std::string& text) const {
    std::vector<int> tokens;
    tokens.reserve(text.size());
    
    for (char c : text) {
        std::string char_str(1, c);
        auto it = token_to_idx.find(char_str);
        if (it != token_to_idx.end()) {
            tokens.push_back(it->second);
        } else {
            tokens.push_back(unk_idx);
        }
    }
    
    return tokens;
}

std::vector<int> Tokenizer::encode_bpe(const std::string& text) const {
    std::vector<std::string> tokens = tokenize_bpe(text);
    std::vector<int> indices;
    indices.reserve(tokens.size());
    
    for (const auto& token : tokens) {
        auto it = token_to_idx.find(token);
        if (it != token_to_idx.end()) {
            indices.push_back(it->second);
        } else {
            indices.push_back(unk_idx);
        }
    }
    
    return indices;
}

std::string Tokenizer::decode_character_level(const std::vector<int>& tokens) const {
    std::string text;
    text.reserve(tokens.size());
    
    for (int idx : tokens) {
        auto it = idx_to_token.find(idx);
        if (it != idx_to_token.end()) {
            const std::string& token = it->second;
            if (token != PAD_TOKEN && token != BOS_TOKEN && token != EOS_TOKEN) {
                text += token;
            }
        } else {
            text += UNK_TOKEN;
        }
    }
    
    return text;
}

std::string Tokenizer::decode_bpe(const std::vector<int>& tokens) const {
    std::string text;
    
    for (int idx : tokens) {
        auto it = idx_to_token.find(idx);
        if (it != idx_to_token.end()) {
            const std::string& token = it->second;
            if (token != PAD_TOKEN && token != BOS_TOKEN && token != EOS_TOKEN) {
                text += token;
            }
        } else {
            text += UNK_TOKEN;
        }
    }
    
    return text;
}

int Tokenizer::vocab_size() const {
    return token_to_idx.size();
}

const std::unordered_map<std::string, int>& Tokenizer::get_token_to_idx() const {
    return token_to_idx;
}

const std::unordered_map<int, std::string>& Tokenizer::get_idx_to_token() const {
    return idx_to_token;
}

std::vector<std::string> Tokenizer::get_bpe_pairs(const std::string& text) const {
    std::vector<std::string> pairs;
    for (size_t i = 0; i < text.length() - 1; ++i) {
        pairs.push_back(text.substr(i, 2));
    }
    return pairs;
}

std::string Tokenizer::merge_bpe_pairs(const std::string& text) const {
    std::string result = text;
    bool merged;
    do {
        merged = false;
        auto pairs = get_bpe_pairs(result);
        for (const auto& pair : pairs) {
            auto it = bpe_merges.find(pair);
            if (it != bpe_merges.end()) {
                // Replace the pair with its merged form
                size_t pos = result.find(pair);
                if (pos != std::string::npos) {
                    std::string merged_token = std::to_string(it->second);
                    result.replace(pos, 2, merged_token);
                    merged = true;
                    break;
                }
            }
        }
    } while (merged);
    return result;
}

void Tokenizer::learn_bpe_merges(const std::vector<std::string>& texts, int target_vocab_size) {
    // Count pair frequencies
    std::unordered_map<std::string, int> pair_freqs;
    for (const auto& text : texts) {
        auto pairs = get_bpe_pairs(text);
        for (const auto& pair : pairs) {
            pair_freqs[pair]++;
        }
    }
    
    // Perform merges until we reach target vocabulary size
    while (token_to_idx.size() < target_vocab_size && !pair_freqs.empty()) {
        // Find most frequent pair
        auto best_pair = std::max_element(pair_freqs.begin(), pair_freqs.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
        
        if (best_pair == pair_freqs.end()) break;
        
        // Add new merged token to vocabulary
        int new_idx = token_to_idx.size();
        token_to_idx[best_pair->first] = new_idx;
        idx_to_token[new_idx] = best_pair->first;
        bpe_merges[best_pair->first] = new_idx;
        
        // Update pair frequencies
        pair_freqs.erase(best_pair);
    }
}

std::vector<std::string> Tokenizer::tokenize_bpe(const std::string& text) const {
    std::vector<std::string> tokens;
    std::string current = text;
    
    while (!current.empty()) {
        // Try to find the longest matching token
        size_t max_len = 0;
        std::string best_token;
        
        for (const auto& [token, _] : token_to_idx) {
            if (current.find(token) == 0 && token.length() > max_len) {
                max_len = token.length();
                best_token = token;
            }
        }
        
        if (max_len > 0) {
            tokens.push_back(best_token);
            current = current.substr(max_len);
        } else {
            // If no match found, use UNK token
            tokens.push_back(UNK_TOKEN);
            current = current.substr(1);
        }
    }
    
    return tokens;
} 