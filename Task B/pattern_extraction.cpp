#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <regex>
#include <map>
#include <algorithm>
#include <nlohmann/json.hpp> // Include the JSON library

using json = nlohmann::json;
using namespace std;

// Structure to store phrase information
struct Phrase {
    int begin;
    int end;
    string text;
    string phrase_type;
};

// Structure to store sentence information
struct Sentence {
    string filename;
    int para_id;
    int sent_id;
    string sent_text;
    vector<Phrase> phrases;
};

// Helper function to split a string by delimiter
vector<string> split(const string &str, char delimiter) {
    vector<string> tokens;
    stringstream ss(str);
    string token;

    while (getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }

    return tokens;
}

// Join strings with a delimiter
string join(const vector<string> &words, const string &delimiter) {
    string result;
    for (int i = 0; i < words.size(); ++i) {
        result += words[i];
        if (i < words.size() - 1) {
            result += delimiter;
        }
    }
    return result;
}

// Function to calculate character offset for phrase position
int calculate_offset(const vector<string> &words_tags, int pos) {
    int offset = 0;
    for (int i = 0; i < pos; ++i) {
        vector<string> split_word_tag = split(words_tags[i], '/');
        if (split_word_tag.size() == 2) {
            offset += split_word_tag[0].size() + 1; // +1 for the space
        }
    }
    return offset;
}

// Extract phrases based on pattern type
vector<Phrase> extract_patterns(const string &raw_text, const string &pattern_type) {
    vector<string> words_tags = split(raw_text, ' ');
    vector<Phrase> extracted_phrases;

    for (int i = 0; i < words_tags.size(); ++i) {
        vector<string> tag_sequence;
        vector<string> word_sequence;

        for (int j = i; j < words_tags.size(); ++j) {
            string word_tag = words_tags[j];
            vector<string> split_word_tag = split(word_tag, '/');

            if (split_word_tag.size() != 2) {
                continue;
            }

            string word = split_word_tag[0];
            string tag = split_word_tag[1];

            tag_sequence.push_back(tag);
            word_sequence.push_back(word);

            // Check for pattern 1: in + nn nn ...
            if (pattern_type == "pattern 1" && tag_sequence[0] == "in" &&
                all_of(tag_sequence.begin() + 1, tag_sequence.end(), [](const string &t) { return t == "nn"; })) {

                string phrase_type = "in " + tag_sequence[1];
                int begin = calculate_offset(words_tags, i);
                int end = calculate_offset(words_tags, j + 1) - 1;

                Phrase phrase = {
                    begin, end,
                    join(word_sequence, " "),
                    phrase_type
                };
                extracted_phrases.push_back(phrase);
            }

            // Check for pattern 2: jj + nn nn ...
            else if (pattern_type == "pattern 2" && tag_sequence[0] == "jj" &&
                     all_of(tag_sequence.begin() + 1, tag_sequence.end(), [](const string &t) { return t == "nn"; })) {

                string phrase_type = "jj " + tag_sequence[1];
                int begin = calculate_offset(words_tags, i);
                int end = calculate_offset(words_tags, j + 1) - 1;

                Phrase phrase = {
                    begin, end,
                    join(word_sequence, " "),
                    phrase_type
                };
                extracted_phrases.push_back(phrase);
            }
        }
    }

    // Return the largest matching phrase if any are found
    if (!extracted_phrases.empty()) {
        auto largest_phrase = max_element(extracted_phrases.begin(), extracted_phrases.end(),
                                          [](const Phrase &a, const Phrase &b) {
                                              return a.text.size() < b.text.size();
                                          });
        return { *largest_phrase }; // Return only the largest phrase
    }

    return {};
}

// Generate JSON output for a pattern
json generate_json_output(vector<vector<string>> &data, const string &pattern_type) {
    json results;
    results["pattern"] = pattern_type;
    results["sents"] = json::array();

    for (const auto &row : data) {
        if (row.size() != 4) {
            continue;
        }

        string filename = row[0];
        int para_id = stoi(row[1]);
        int sent_id = stoi(row[2]);
        string raw_text = row[3];

        vector<Phrase> phrases = extract_patterns(raw_text, pattern_type);

        if (!phrases.empty()) {
            vector<string> words;
            for (const auto &word_tag : split(raw_text, ' ')) {
                vector<string> split_word_tag = split(word_tag, '/');
                if (split_word_tag.size() == 2) {
                    words.push_back(split_word_tag[0]);
                }
            }

            string sent_text = join(words, " ");

            json sent_json;
            sent_json["filename"] = filename;
            sent_json["para_id"] = para_id;
            sent_json["sent_id"] = sent_id;
            sent_json["sent_text"] = sent_text;
            sent_json["phrases"] = json::array();

            for (const auto &phrase : phrases) {
                json phrase_json;
                phrase_json["begin"] = phrase.begin;
                phrase_json["end"] = phrase.end;
                phrase_json["text"] = phrase.text;
                phrase_json["phrase_type"] = phrase.phrase_type;

                sent_json["phrases"].push_back(phrase_json);
            }

            results["sents"].push_back(sent_json);
        }
    }

    return results;
}

// Load CSV data into a vector of rows
vector<vector<string>> load_csv(const string &filename) {
    vector<vector<string>> data;
    ifstream file(filename);
    string line;

    // Skip header line
    getline(file, line);

    while (getline(file, line)) {
        vector<string> row;
        stringstream ss(line);
        string cell;

        while (getline(ss, cell, ',')) {
            row.push_back(cell);
        }

        if (row.size() == 4) {
            data.push_back(row);
        }
    }

    return data;
}

// Save JSON results to a file
void save_json_to_file(const json &results, const string &filename) {
    ofstream file(filename);
    if (file.is_open()) {
        file << results.dump(4); // Indent by 4 spaces for readability
        file.close();
    } else {
        cerr << "Unable to open file: " << filename << endl;
    }
}

// Main function
int main() {
    string csv_file = "Task_B_Dataset.csv";
    vector<vector<string>> data = load_csv(csv_file);

    // Generate and save Pattern 1 results
    json pattern1_results = generate_json_output(data, "pattern 1");
    save_json_to_file(pattern1_results, "pattern1_results.json");

    // Generate and save Pattern 2 results
    json pattern2_results = generate_json_output(data, "pattern 2");
    save_json_to_file(pattern2_results, "pattern2_results.json");

    cout << "Pattern extraction and JSON generation complete!" << endl;

    return 0;
}