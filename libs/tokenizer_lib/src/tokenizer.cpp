#include <tokenizer_lib/tokenizer.h>
#include <algorithm>
#include <set>
#include <queue>
#include <sstream>
#include <iostream>

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

    fit_character_level(texts);
    
    // Learn BPE merges
    learn_bpe_merges(texts, vocab_size);
}

std::vector<std::vector<int>> Tokenizer::encode_character_level(const std::vector<std::string>& texts) const {
    std::vector<std::vector<int>> encoded_texts;
    encoded_texts.reserve(texts.size());
    
    for (const auto& text : texts) {
        std::vector<int> tokens;
        tokens.reserve(text.size() + 2);  // +2 for BOS and EOS tokens
        
        // Add BOS token
        tokens.push_back(bos_idx);
        
        // Encode each character
        for (char c : text) {
            std::string char_str(1, c);
            auto it = token_to_idx.find(char_str);
            if (it != token_to_idx.end()) {
                tokens.push_back(it->second);
            } else {
                tokens.push_back(unk_idx);
            }
        }
        
        // Add EOS token
        tokens.push_back(eos_idx);
        
        encoded_texts.push_back(std::move(tokens));
    }
    
    return encoded_texts;
}

std::vector<std::vector<int>> Tokenizer::encode_bpe(const std::vector<std::string>& texts) const {
    std::vector<std::vector<int>> encoded_texts;
    encoded_texts.reserve(texts.size());
    
    for (const auto& text : texts) {
        std::vector<std::string> tokens = tokenize_bpe(text);
        std::vector<int> indices;
        indices.reserve(tokens.size() + 2);  // +2 for BOS and EOS tokens
        
        // Add BOS token
        indices.push_back(bos_idx);
        
        // Convert tokens to indices
        for (const auto& token : tokens) {
            auto it = token_to_idx.find(token);
            if (it != token_to_idx.end()) {
                indices.push_back(it->second);
            } else {
                indices.push_back(unk_idx);
            }
        }
        
        // Add EOS token
        indices.push_back(eos_idx);
        
        encoded_texts.push_back(std::move(indices));
    }
    
    return encoded_texts;
}

std::string Tokenizer::decode_character_level(const std::vector<int>& tokens) const {
    std::string text = "";
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

void Tokenizer::learn_bpe_merges(const std::vector<std::string>& texts, int target_vocab_size) {
    
    std::vector<std::vector<std::string>> tokenized_texts;
    
    for(int i=0; i<texts.size(); i++){
        tokenized_texts.push_back(std::vector<std::string>());
        for(int j=0; j<texts[i].size(); j++){
            tokenized_texts[i].push_back(std::string(1, texts[i][j]));
        }
    }
    
    int idx = token_to_idx.size();

    std::unordered_map<std::string, int> pair_freqs;
    while (token_to_idx.size() < target_vocab_size){
        pair_freqs.clear();
        for(int i=0; i<tokenized_texts.size(); i++){
            for(int j=0; j<tokenized_texts[i].size()-1; j++){
                std::string pair = tokenized_texts[i][j] + tokenized_texts[i][j+1];
                pair_freqs[pair]++;
            }
        }
        if(pair_freqs.empty()) break;
        auto best_pair = std::max_element(pair_freqs.begin(), pair_freqs.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
            
        token_to_idx[best_pair->first] = idx;
        idx_to_token[idx] = best_pair->first;
        idx++;

        for(int i=0; i<tokenized_texts.size(); i++){
            for(int j=0; j<tokenized_texts[i].size()-1; j++){
                if(tokenized_texts[i][j] + tokenized_texts[i][j+1] == best_pair->first){
                    tokenized_texts[i][j] = best_pair->first;
                    tokenized_texts[i].erase(tokenized_texts[i].begin() + j + 1);
                    j--;
                }
            }
        }
    }

    // // Count pair frequencies
    // std::unordered_map<std::string, int> pair_freqs;
    // for (const auto& text : texts) {
    //     auto pairs = get_bpe_pairs(text);
    //     for (const auto& pair : pairs) {
    //         pair_freqs[pair]++;
    //     }
    // }
    
    // // Perform merges until we reach target vocabulary size
    // while (token_to_idx.size() < target_vocab_size && !pair_freqs.empty()) {
    //     // Find most frequent pair
    //     auto best_pair = std::max_element(pair_freqs.begin(), pair_freqs.end(),
    //         [](const auto& a, const auto& b) { return a.second < b.second; });
        
    //     if (best_pair == pair_freqs.end()) break;
        
    //     // Add new merged token to vocabulary
    //     int new_idx = token_to_idx.size();
    //     token_to_idx[best_pair->first] = new_idx;
    //     idx_to_token[new_idx] = best_pair->first;
    //     bpe_merges[best_pair->first] = new_idx;
        
    //     // Update pair frequencies
    //     pair_freqs.erase(best_pair);
    // }
}

std::vector<std::string> Tokenizer::tokenize_bpe(const std::string& text) const {
    std::vector<std::string> tokens;

    for(int i=0; i<text.size(); i++){
        tokens.push_back(std::string(1, text[i]));
    }

    bool flag=true;
    while (flag) {
        flag=false;

        for(int i=0; i<tokens.size()-1; i++){
            std::string pair = tokens[i] + tokens[i+1];
            auto it = token_to_idx.find(pair);
            if(it != token_to_idx.end()){
                tokens[i] = pair;
                tokens.erase(tokens.begin() + i + 1);
                flag=true;
                break;
            }
        }
    }
    
    return tokens;
}