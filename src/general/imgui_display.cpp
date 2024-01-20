//
// Created by Palnit on 2024. 01. 20.
//

#include "include/general/imgui_display.h"
#include "include/general/main_window.h"
#include <imgui.h>

void ImGuiDisplay::DisplayImGui() {
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImVec2(m_width / 3, m_height));
    if (!ImGui::Begin("Edge Detector Options",
                      &m_open,
                      ImGuiWindowFlags_NoMove
                          | ImGuiWindowFlags_NoDocking
                          | ImGuiWindowFlags_NoResize)) {
        ImGui::End();
        return;
    }
    ImGui::RadioButton("Canny CPU", &m_add, 0);
    ImGui::SameLine();
    ImGui::RadioButton("Canny GPU", &m_add, 1);
    ImGui::RadioButton("DOG CPU", &m_add, 2);
    ImGui::SameLine();
    ImGui::RadioButton("DOG GPU", &m_add, 3);

    ImGui::InputText("Detector Name", m_buf, 300);

    ImGui::ListBox("Choose picture",
                   &m_picture,
                   VectorOfStringGetter,
                   (void*) &m_pictures,
                   (int) m_pictures.size());

    if (ImGui::Button("Add")) {
        auto* parent = dynamic_cast<MainWindow*>(m_parent);
        std::string file = "pictures/" + m_pictures.at(m_picture);
        SDL_Surface* m_base = FileHandling::LoadImage(file.c_str());

        m_detectors.push_back(new DetectorCuda(m_base, m_buf));
        parent->AddDetector(m_detectors.back());
        m_names.emplace_back(m_buf);
    }

    ImGui::ListBox("Detectors",
                   &m_remove,
                   VectorOfStringGetter,
                   (void*) &m_names,
                   (int) m_names.size());

    if (ImGui::Button("Remove")) {
        auto* parent = dynamic_cast<MainWindow*>(m_parent);
        parent->RemoveDetector(m_detectors.at(m_remove));
        m_detectors.erase(m_detectors.begin() + m_remove);
        m_names.erase(m_names.begin() + m_remove);
    }
    ImGui::Separator();
    if (ImGui::BeginTabBar("Detector Options")) {
        for (DetectorBase* detector : m_detectors) {
            detector->DisplayImGui();
        }
        ImGui::EndTabBar();
    }
    ImGui::End();

}
