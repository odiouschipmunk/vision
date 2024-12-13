#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

struct Keypoint {
    double x;
    double y;
};

struct FrameData {
    int frameCount;
    std::vector<Keypoint> player1Keypoints;
    std::vector<Keypoint> player2Keypoints;
    std::pair<int, int> ballPosition;
    std::string shotType;
};

std::vector<FrameData> parseCSV(const std::string& filename) {
    std::vector<FrameData> data;
    std::ifstream file(filename);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return data;
    }

    // Skip the header
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string token;
        FrameData frameData;

        // Parse frame count
        std::getline(ss, token, ',');
        frameData.frameCount = std::stoi(token);

        // Parse player 1 keypoints
        std::getline(ss, token, ',');
        std::istringstream p1ss(token.substr(1, token.size() - 2));
        std::string p1keypoint;
        while (std::getline(p1ss, p1keypoint, ']')) {
            if (p1keypoint.size() > 1) {
                std::istringstream kp(p1keypoint.substr(1));
                std::string x, y;
                std::getline(kp, x, ',');
                std::getline(kp, y, ',');
                frameData.player1Keypoints.push_back({std::stod(x), std::stod(y)});
            }
        }

        // Parse player 2 keypoints
        std::getline(ss, token, ',');
        std::istringstream p2ss(token.substr(1, token.size() - 2));
        std::string p2keypoint;
        while (std::getline(p2ss, p2keypoint, ']')) {
            if (p2keypoint.size() > 1) {
                std::istringstream kp(p2keypoint.substr(1));
                std::string x, y;
                std::getline(kp, x, ',');
                std::getline(kp, y, ',');
                frameData.player2Keypoints.push_back({std::stod(x), std::stod(y)});
            }
        }

        // Parse ball position
        std::getline(ss, token, ',');
        std::istringstream bss(token.substr(1, token.size() - 2));
        std::string bx, by;
        std::getline(bss, bx, ',');
        std::getline(bss, by, ',');
        frameData.ballPosition = {std::stoi(bx), std::stoi(by)};

        // Parse shot type
        std::getline(ss, token, ',');
        frameData.shotType = token;

        data.push_back(frameData);
    }

    file.close();
    return data;
}

int main() {
    std::string filename = "output/final.csv";
    std::vector<FrameData> data = parseCSV(filename);

    // Example usage: print the parsed data
    for (const auto& frame : data) {
        std::cout << "Frame Count: " << frame.frameCount << std::endl;
        std::cout << "Player 1 Keypoints: ";
        for (const auto& kp : frame.player1Keypoints) {
            std::cout << "[" << kp.x << ", " << kp.y << "] ";
        }
        std::cout << std::endl;

        std::cout << "Player 2 Keypoints: ";
        for (const auto& kp : frame.player2Keypoints) {
            std::cout << "[" << kp.x << ", " << kp.y << "] ";
        }
        std::cout << std::endl;

        std::cout << "Ball Position: [" << frame.ballPosition.first << ", " << frame.ballPosition.second << "]" << std::endl;
        std::cout << "Shot Type: " << frame.shotType << std::endl;
        std::cout << "---------------------------------" << std::endl;
    }

    return 0;
}