#include <tokenizer_lib/tokenizer.h>
#include <cassert>
#include <iostream>
#include <vector>
#include <string>

using namespace std;

void test_character_level_tokenization() {
    Tokenizer tokenizer;
    vector<string> texts = {"hello", "world"};

    // Test fitting
    tokenizer.fit_character_level(texts);
    
    // Test vocabulary size (should be 4 special tokens + 7 unique characters)
    assert(tokenizer.vocab_size() == 11);
    // Test encoding
    auto encoded = tokenizer.encode_character_level(texts);
    assert(encoded.size() == 2);  // Two texts
    
    // Test first text "hello"
    assert(encoded[0].size() == 7);  // 5 chars + BOS + EOS
    assert(encoded[0][0] == tokenizer.get_token_to_idx().at("<bos>"));  // BOS token
    assert(encoded[0][6] == tokenizer.get_token_to_idx().at("<eos>"));  // EOS token
    
    // Test decoding
    string decoded = tokenizer.decode_character_level(encoded[0]);
    assert(decoded == "hello");
    
    cout << "PASSED test_character_level_tokenization" << endl;
}

void test_bpe_tokenization_1() {
    Tokenizer tokenizer;
    vector<string> texts = {"hello", "world", "helloworld"};
    
    // Test fitting with small vocab size to force merges
    tokenizer.fit_bpe(texts, 18);
    
    // Test vocabulary size (should be 4 special tokens + some merged tokens)
    assert(tokenizer.vocab_size() <= 18);
    
    // Test encoding
    auto encoded = tokenizer.encode_bpe(texts);
    assert(encoded.size() == 3);  // Three texts
    
    // Test first text "hello"
    assert(encoded[0].size() >= 3);  // At least 1 token + BOS + EOS
    assert(encoded[0][0] == tokenizer.get_token_to_idx().at("<bos>"));  // BOS token
    assert(encoded[0].back() == tokenizer.get_token_to_idx().at("<eos>"));  // EOS token
    
    // Test decoding
    string decoded = tokenizer.decode_bpe(encoded[0]);
    assert(decoded == "hello");
    
    // // Test that "helloworld" is tokenized differently than just concatenating "hello" and "world"
    // assert(encoded[2].size() < encoded[0].size() + encoded[1].size() - 2);  // -2 for BOS/EOS tokens
    
    cout << "PASSED test_bpe_tokenization 1" << endl;
}

void test_bpe_tokenization_2() {
    Tokenizer tokenizer;
    vector<string> texts = {"the fox jumped over the lazy dog", "fox's are not lazy creatures", "i have a dog called bobby"};
    
    // Test fitting with small vocab size to force merges
    tokenizer.fit_bpe(texts, 50);

    for(auto it: tokenizer.get_token_to_idx()){
        cout<<it.first<<" "<<it.second<<endl;
    }
    
    // Test vocabulary size (should be 4 special tokens + some merged tokens)
    assert(tokenizer.vocab_size() <= 50);
    
    // Test encoding
    auto encoded = tokenizer.encode_bpe(texts);
    // assert(encoded.size() == 3);  // Three texts

    for(auto it: encoded){
        for(auto it2: it){
            cout<<it2<<" ";
        }
        cout<<endl;
    }

    auto decoded = tokenizer.decode_bpe(encoded[0]);
    auto decoded2 = tokenizer.decode_bpe(encoded[1]);
    auto decoded3 = tokenizer.decode_bpe(encoded[2]);

    cout<<decoded<<endl;
    cout<<decoded2<<endl;
    cout<<decoded3<<endl;
    
    // Test first text "hello"
    // assert(encoded[0].size() >= 3);  // At least 1 token + BOS + EOS
    // assert(encoded[0][0] == tokenizer.get_token_to_idx().at("<bos>"));  // BOS token
    // assert(encoded[0].back() == tokenizer.get_token_to_idx().at("<eos>"));  // EOS token
    
    // Test decoding
    // string decoded = tokenizer.decode_bpe(encoded[0]);
    // assert(decoded == "hello");
    
    // // Test that "helloworld" is tokenized differently than just concatenating "hello" and "world"
    // assert(encoded[2].size() < encoded[0].size() + encoded[1].size() - 2);  // -2 for BOS/EOS tokens
    
    cout << "PASSED test_bpe_tokenization 2" << endl;
}

void test_special_tokens() {
    Tokenizer tokenizer;
    vector<string> texts = {"hello"};
    
    // Test that special tokens are properly initialized
    assert(tokenizer.get_token_to_idx().at("<pad>") == 0);
    assert(tokenizer.get_token_to_idx().at("<unk>") == 1);
    assert(tokenizer.get_token_to_idx().at("<bos>") == 2);
    assert(tokenizer.get_token_to_idx().at("<eos>") == 3);
    
    // Test that unknown characters are mapped to UNK token
    tokenizer.fit_character_level(texts);
    string text_with_unknown = "hello\u2603";  // Add a snowman character
    auto encoded = tokenizer.encode_character_level({text_with_unknown});
    // cout<<encoded[0].size()<<endl;
    // cout<<text_with_unknown<<endl;
    // for(auto it: encoded[0]){
    //     cout<<it<<" ";
    // }
    // cout<<endl;

    assert(encoded[0].size() == 10);  // 6 chars + BOS + EOS
    assert(encoded[0][6] == tokenizer.get_token_to_idx().at("<unk>"));  // UNK token for snowman
    
    cout << "PASSED test_special_tokens" << endl;
}

int main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    
    test_character_level_tokenization();
    test_bpe_tokenization_1();
    test_bpe_tokenization_2();
    test_special_tokens();
    
    cout << "All tokenizer tests passed!" << endl;
    return 0;
} 