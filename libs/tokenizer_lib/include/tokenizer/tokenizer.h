#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <set>
#include <utility>

class Tokenizer {
public:
    // Constructor
    Tokenizer();
    
    // Fit methods for different tokenization strategies
    void fit_character_level(const std::vector<std::string>& texts);
    void fit_bpe(const std::vector<std::string>& texts, int vocab_size);
    
    // Encode methods for different tokenization strategies
    std::vector<int> encode_character_level(const std::string& text) const;
    std::vector<int> encode_bpe(const std::string& text) const;
    
    // Decode methods for different tokenization strategies
    std::string decode_character_level(const std::vector<int>& tokens) const;
    std::string decode_bpe(const std::vector<int>& tokens) const;
    
    // Get vocabulary size
    int vocab_size() const;
    
    // Get the mappings
    const std::unordered_map<std::string, int>& get_token_to_idx() const;
    const std::unordered_map<int, std::string>& get_idx_to_token() const;

private:
    // Maps tokens to their indices
    std::unordered_map<std::string, int> token_to_idx;
    
    // Maps indices back to tokens
    std::unordered_map<int, std::string> idx_to_token;
    
    // Special tokens
    static constexpr const char* PAD_TOKEN = "<pad>";
    static constexpr const char* UNK_TOKEN = "<unk>";
    static constexpr const char* BOS_TOKEN = "<bos>";
    static constexpr const char* EOS_TOKEN = "<eos>";
    
    // Special token indices
    int pad_idx;
    int unk_idx;
    int bos_idx;
    int eos_idx;
    
    // BPE specific members
    std::unordered_map<std::string, int> bpe_merges;
    
    // Helper methods for BPE
    std::vector<std::string> get_bpe_pairs(const std::string& text) const;
    std::string merge_bpe_pairs(const std::string& text) const;
    void learn_bpe_merges(const std::vector<std::string>& texts, int target_vocab_size);
    std::vector<std::string> tokenize_bpe(const std::string& text) const;
};

#endif // TOKENIZER_H 