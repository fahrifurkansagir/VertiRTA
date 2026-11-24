// SPDX-License-Identifier: Apache-2.0
#include "vertirta/forensics/eccairs2_e5x_emitter.hpp"

#include <algorithm>
#include <sstream>
#include <vector>

#include "vertirta/forensics/narrative_grounding.hpp"

namespace vertirta::forensics {

namespace {

struct XmlAttributeRow {
  uint32_t id{0U};
  std::string name{};
  std::string value{};
};

std::string xml_escape(const std::string& raw) {
  std::string out;
  out.reserve(raw.size() + 16U);
  for (const char ch : raw) {
    switch (ch) {
      case '&':
        out += "&amp;";
        break;
      case '<':
        out += "&lt;";
        break;
      case '>':
        out += "&gt;";
        break;
      case '"':
        out += "&quot;";
        break;
      case '\'':
        out += "&apos;";
        break;
      default:
        out.push_back(ch);
        break;
    }
  }
  return out;
}

void write_attributes_sorted(
    std::ostringstream& oss,
    const std::string& indent,
    std::vector<XmlAttributeRow> rows) {
  std::sort(rows.begin(), rows.end(), [](const XmlAttributeRow& a, const XmlAttributeRow& b) {
    return a.id < b.id;
  });
  oss << indent << "<ATTRIBUTES>\n";
  uint32_t last_id = 0U;
  bool has_last = false;
  for (const auto& row : rows) {
    if (has_last && row.id == last_id) {
      continue;  // Guarantee strictly ascending unique attribute IDs
    }
    has_last = true;
    last_id = row.id;
    oss << indent << "  <Attribute id=\"" << row.id << "\" name=\"" << xml_escape(row.name)
        << "\">" << xml_escape(row.value) << "</Attribute>\n";
  }
  oss << indent << "</ATTRIBUTES>\n";
}

}  // namespace

std::string emit_eccairs2_e5x_xml(const E5xOccurrenceDraft& draft) {
  // Always apply Regulation (EU) 376/2014 Article 16 de-identification to Headline ID 601.
  const std::string safe_headline = sanitize_article16_headline(draft.headline_raw);
  const std::string draft_class_str =
      (draft.ercs.occurrence_class_id_431 == OccurrenceClassDraft::DRAFT_NOT_DETERMINED)
          ? "NOT_DETERMINED_DRAFT_ONLY"
          : "INCIDENT_DRAFT_ONLY";

  // Note: N/A injury count attributes (114, 187, 262, 460, 462, 463, 469, 472) are
  // intentionally omitted (never zero-filled when N/A per ECCAIRS2/ADREP rules).
  std::vector<XmlAttributeRow> occ_attrs = {
      {16U, "FIR_UIR", draft.fir_uir_id_16},
      {121U, "EVTOL_ROTORCRAFT_FLIGHT_PHASE", std::to_string(draft.evtol_flight_phase_id_121)},
      {372U, "ATS_UNIT", draft.ats_unit_id_372},
      {430U, "OCCURRENCE_CATEGORY_CICTT", draft.ercs.cictt_category},
      {431U, "OCCURRENCE_CLASS_DRAFT", draft_class_str},
      {433U, "LOCAL_DATE_TIME", draft.local_date_time_id_433},
      {440U, "LOCATION_NAME", draft.location_name_id_440},
      {452U, "FILE_NUMBER", draft.file_number},
      {453U, "REPORTING_ENTITY", draft.reporting_entity},
      {454U, "STATE_AREA_OF_OCCURRENCE", draft.state_area_id_454},
      {477U, "UTC_DATE_TIME", draft.utc_date_time_id_477},
      {601U, "HEADLINE", safe_headline},
      {659U, "ATA_JASC_CHAPTER", std::to_string(draft.ercs.ata_jasc_chapter)},
      {1109U, "ERCS_RISK_GRADE", draft.ercs.ercs_matrix_code},
  };

  std::vector<XmlAttributeRow> aircraft_attrs = {
      {21U, "AIRCRAFT_MODEL", draft.aircraft_model},
      {121U, "FLIGHT_PHASE", std::to_string(draft.evtol_flight_phase_id_121)},
      {175U, "MTOW_KG", std::to_string(draft.mtow_kg)},
  };

  std::vector<XmlAttributeRow> event_attrs = {
      {390U, "EVENT_TYPE_ADREP", draft.ercs.cictt_category},
      {391U, "EVENT_PHASE", std::to_string(draft.evtol_flight_phase_id_121)},
  };

  std::vector<XmlAttributeRow> part_attrs = {
      {659U, "ATA_JASC_CHAPTER", std::to_string(draft.ercs.ata_jasc_chapter)},
  };

  std::ostringstream oss;
  oss << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
  oss << "<SET xmlns=\"http://eccairsportal.jrc.ec.europa.eu/schemas/e5x-minimal/v1\" "
         "schemaVersion=\"vertirta.e5x.draft/v1.0.0\">\n";
  oss << "  <Occurrence entityId=\"24\" reportingStatus=\"DRAFT_INTERNAL_PRE_SUBMISSION\" "
         "complianceClaim=\"NONE_RESEARCH_TESTBED_ONLY\">\n";
  write_attributes_sorted(oss, "    ", std::move(occ_attrs));

  oss << "    <METADATA>\n";
  oss << "      <Cir2026Trigger>" << xml_escape(to_string(draft.cir_trigger))
      << "</Cir2026Trigger>\n";
  oss << "      <ShgmShyOlayDeadlineHours>" << draft.shgm_deadline_hours
      << "</ShgmShyOlayDeadlineHours>\n";
  oss << "      <VrtaFileSha256>" << xml_escape(draft.vrta_file_sha256) << "</VrtaFileSha256>\n";
  oss << "      <SidecarChainVerified>" << (draft.sidecar_chain_verified ? "true" : "false")
      << "</SidecarChainVerified>\n";
  oss << "      <NumericGroundingVerified>"
      << (draft.numeric_grounding_verified ? "true" : "false") << "</NumericGroundingVerified>\n";
  oss << "    </METADATA>\n";

  oss << "    <ENTITIES>\n";
  oss << "      <Aircraft entityId=\"4\">\n";
  write_attributes_sorted(oss, "        ", std::move(aircraft_attrs));
  oss << "      </Aircraft>\n";
  oss << "      <Event entityId=\"14\">\n";
  write_attributes_sorted(oss, "        ", std::move(event_attrs));
  oss << "      </Event>\n";
  oss << "      <Narrative entityId=\"22\" language=\"en\">" << xml_escape(draft.narrative_en)
      << "</Narrative>\n";
  oss << "      <Narrative entityId=\"22\" language=\"tr\">" << xml_escape(draft.narrative_tr)
      << "</Narrative>\n";
  oss << "      <PartInfo entityId=\"26\">\n";
  write_attributes_sorted(oss, "        ", std::move(part_attrs));
  oss << "      </PartInfo>\n";
  oss << "    </ENTITIES>\n";

  oss << "  </Occurrence>\n";
  oss << "</SET>\n";
  return oss.str();
}

}  // namespace vertirta::forensics
