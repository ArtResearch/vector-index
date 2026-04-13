#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <filesystem>
#include <fstream>
#include <omp.h>
#include <boost/program_options.hpp>
#include "lazycsv.hpp"

namespace po = boost::program_options;
namespace fs = std::filesystem;

struct string_hash {
    using is_transparent = void;
    [[nodiscard]] size_t operator()(const char* txt) const {
        return std::hash<std::string_view>{}(txt);
    }
    [[nodiscard]] size_t operator()(std::string_view txt) const {
        return std::hash<std::string_view>{}(txt);
    }
    [[nodiscard]] size_t operator()(const std::string& txt) const {
        return std::hash<std::string>{}(txt);
    }
};

void append_escaped(std::string& dest, const std::string& src) {
    dest += '"';
    for (char c : src) {
        if (c == '"') {
            dest += "\"\"";
        } else {
            dest += c;
        }
    }
    dest += '"';
}

int main(int argc, char* argv[]) {
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message")
        ("data-dir", po::value<std::string>()->required(), "directory with input csv files")
        ("works-csv", po::value<std::string>()->required(), "path to works csv file")
        ("output-csv", po::value<std::string>()->required(), "path to output csv file")
        ("max-cpu-threads", po::value<int>()->default_value(20), "max cpu threads to use");

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);

        if (vm.count("help")) {
            std::cout << desc << std::endl;
            return 1;
        }

        po::notify(vm);
    } catch (const po::error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::cerr << desc << std::endl;
        return 1;
    }

    std::string current_filepath;
    try {
        std::string data_dir = vm["data-dir"].as<std::string>();
        std::string works_csv_path = vm["works-csv"].as<std::string>();
        std::string output_csv_path = vm["output-csv"].as<std::string>();
        int max_cpu_threads = vm["max-cpu-threads"].as<int>();

        // Set the number of threads for OpenMP
        omp_set_num_threads(max_cpu_threads);

        // Read the initial list of URIs and other data
        lazycsv::parser<lazycsv::mmap_source, lazycsv::has_header<true>> works_parser(works_csv_path);
        auto header = works_parser.header();
        std::vector<std::string> output_header;
        int num_columns = 0;
        for(const auto& h : header) {
            output_header.push_back(h.unescaped());
            num_columns++;
        }
        output_header.push_back("text");

        int subject_idx = works_parser.index_of("uri");
        std::vector<std::string> uris;
        std::vector<std::vector<std::string>> work_rows_data;
        for (const auto& row : works_parser) {
            uris.push_back(row.cells(subject_idx)[0].unescaped());
            std::vector<std::string> row_data;
            row_data.reserve(num_columns);
            for(const auto& cell : row) {
                row_data.push_back(cell.unescaped());
            }
            work_rows_data.push_back(row_data);
        }
        
        // Use a map for efficient lookups and aggregation
        std::unordered_map<std::string, std::string, string_hash, std::equal_to<>> uri_to_text;
        uri_to_text.reserve(uris.size());
        for (const auto& uri : uris) {
            uri_to_text[uri] = "";
        }

        // Get a list of files to process
        std::vector<fs::path> files_to_process;
        for (const auto& entry : fs::directory_iterator(data_dir)) {
            if (entry.is_regular_file()) {
                files_to_process.push_back(entry.path());
            }
        }

        // Process files in parallel
        #pragma omp parallel
        {
            std::unordered_map<std::string, std::string, string_hash, std::equal_to<>> local_uri_to_text;
            for (const auto& uri : uris) {
                local_uri_to_text[uri] = "";
            }

            #pragma omp for nowait
            for (size_t i = 0; i < files_to_process.size(); ++i) {
                const auto& filepath = files_to_process[i];
                std::string current_filepath_str = filepath.string();
                std::string field = filepath.stem().string();
                
                lazycsv::parser<lazycsv::mmap_source, lazycsv::has_header<true>> parser(current_filepath_str);
                int subject_idx = parser.index_of("subject");
                int value_idx = parser.index_of("value");

                std::string last_uri;
                std::string accumulated_values;

                for (const auto& row : parser) {
                    auto cells = row.cells(subject_idx, value_idx);
                    std::string_view current_uri_sv = cells[0].trimmed();
                    std::string_view current_value_sv = cells[1].trimmed();

                    if (local_uri_to_text.find(current_uri_sv) == local_uri_to_text.end()) {
                        continue;
                    }

                    if (!last_uri.empty() && current_uri_sv != last_uri) {
                        if (!accumulated_values.empty()) {
                            std::string& text = local_uri_to_text[last_uri];
                            text.reserve(text.length() + field.length() + accumulated_values.length() + 4);
                            text.append(field).append(": ").append(accumulated_values).append("; ");
                            accumulated_values.clear();
                        }
                    }

                    if (!accumulated_values.empty()) {
                        accumulated_values += ", ";
                    }
                    accumulated_values.append(current_value_sv);
                    
                    if (last_uri != current_uri_sv) {
                        last_uri = current_uri_sv;
                    }
                }

                // Write the last accumulated values
                if (!accumulated_values.empty()) {
                    std::string& text = local_uri_to_text[last_uri];
                    text.reserve(text.length() + field.length() + accumulated_values.length() + 3);
                    text.append(field).append(":").append(accumulated_values).append("; ");
                }
            }

            // Merge local results into the global map
            #pragma omp critical
            {
                for (const auto& pair : local_uri_to_text) {
                    if (!pair.second.empty()) {
                        uri_to_text[pair.first] += pair.second;
                    }
                }
            }
        }

        // Write the aggregated data to the output CSV
        std::ofstream output_file(output_csv_path);
        
        std::string header_line;
        for(size_t i = 0; i < output_header.size(); ++i) {
            append_escaped(header_line, output_header[i]);
            if (i < output_header.size() - 1) {
                header_line += ',';
            }
        }
        header_line += '\n';
        output_file << header_line;

        std::string buffer;
        constexpr size_t buffer_size = 64 * 1024; // 64KB buffer
        buffer.reserve(buffer_size);
        std::string line;

        for (size_t i = 0; i < work_rows_data.size(); ++i) {
            line.clear();
            const auto& row_data = work_rows_data[i];
            const auto& uri = uris[i];

            for(size_t j = 0; j < row_data.size(); ++j) {
                append_escaped(line, row_data[j]);
                if (j < row_data.size() - 1) {
                    line += ',';
                }
            }

            line += ',';
            const std::string& text = uri_to_text.at(uri);
            append_escaped(line, text);
            line += '\n';
            
            if (buffer.length() + line.length() > buffer_size) {
                output_file.write(buffer.c_str(), buffer.length());
                buffer.clear();
            }
            buffer.append(line);
        }

        if (!buffer.empty()) {
            output_file.write(buffer.c_str(), buffer.length());
        }

        std::cout << "Successfully created " << output_csv_path << std::endl;
        std::quick_exit(0);
    } catch (const std::exception& e) {
        std::cerr << "An error occurred";
        if (!current_filepath.empty()) {
            std::cerr << " in file " << current_filepath;
        }
        std::cerr << ": " << e.what() << std::endl;
        return 1;
    }
}
