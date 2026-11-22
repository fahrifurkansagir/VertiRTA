# DR-7: VertiRTA-X8-240 İçin Sıfır-ShareAlike Apache-2.0 Temel Model Damıtmalı, Aktif Belirsizlik Güdümlü 3B Güvenli İniş Bölgesi (SLZ) Algılama ve Colab-G4 Gazebo Harmonic EGL Mimarisi

> **Belge Kimliği:** `VertiRTA / DR-7`  
> **Platform:** `VertiRTA-X8-240` (240 kg eş eksenli X8 eVTOL)  
> **Fiziksel İniş Zarfı:** Dairesel rotor açıklığı $R_{\text{SLZ}} \ge 5.0$ m, azami eğim $\theta \le 8$ derece, azami yüzey pürüzlülüğü $\sigma_z \le 0.15$ m, azami basamak $\text{step}_{\max} \le 0.25$ m  
> **Operasyon Koridoru:** İstanbul Maltepe – Dragos – Kartal Kıyı Şeridi (`LTFJ` ve `LTBX` CTR duvarlarının batısı, 6.64 km)  
> **Lisans Kilidi:** %100 Temiz `Apache-2.0` ağırlık dağıtımı (Sıfır-ShareAlike, Sıfır-NC, Savunma/Ticari Uyumlu)

---

## Yönetici Özeti

**`VertiRTA-X8-240`** platformu için geliştirilen bu algılama ve aktif karar mimarisi, saf C++20 **`ISlzPriorProvider`** portunu 20–30+ Hz hızında besleyen, 2.5B Bayesçi SLZ İnanç Izgarası tabanlı ve deterministik ~150 satırlık **Aktif İnceleme Yörüngesi (Active Inspection Orbit)** politikasıyla bütünleşik bir uçtan uca çözüm sunar.

2025–2026 birincil kaynak lisans denetimi sonucunda ticari/savunma ağırlık dağıtımını ihlal eden dört popüler aday veri seti — **`LoveDA`** (yalnızca akademik kullanım + Google Earth koşulları), **`RescueNet`** (resmi BinaLab deposunda `CC BY-NC-ND 4.0`; yalnızca makale metni `CC BY 4.0`), **`DroneDeploy`** (lisans dosyası yok; varsayılan tüm hakları saklı) ve **`FlyAwareV2`** (`GPL-3.0` + yasaklı `UAVid` içeriği) — eğitim manifestosundan kesin olarak çıkarılmıştır. **`OpenEarthMap`** içinde ise kaynak görüntüsü Kamu Malı (5 girdi) veya lisansı belirtilmemiş (12 girdi) bölgeler otomatik olarak `CC BY-NC-SA 4.0` kapsamına düştüğünden, 100 kaynak girdisinden yalnızca açıkça izinli **41 girdi** (`CC BY 4.0` + `DL-DE-BY-2.0`) beyaz listeye alınmıştır. Bu açığı kapatmak amacıyla **`FLAIR-1`** (`OPEN LICENCE 2.0 / etalab-2.0`, `CC-BY` uyumlu), **`FloodNet-Supervised v1.0`** (`CDLA-Permissive-1.0`), **`SkyScenes`** (`MIT`, ECCV 2024) ve yazar mülkiyetindeki **8.000 karelik Colab-EGL Gazebo Harmonic İstanbul** veri setinden oluşan 5 kaynaklı tamamen temiz bir eğitim manifestosu sabitlenmiştir.

Gerçek zamanlı öğrenci ağı olarak `512x512` çözünürlükte sınır keskinliği, tek alt-graf TensorRT 10 FP16 uyumluluğu ve 300 MB tavanının %37 altında kalan ~189.0 MB tahminî FP16 VRAM bütçesiyle **`PIDNet-S` (MIT)** tek kazanan mimari seçilmiştir. Model, Colab G4 (`RTX PRO 6000 Blackwell`, 96 GB VRAM) üzerinde dondurulmuş **`DINOv2-ViT-L/14` (Apache-2.0)** yoğun yama öznitelikleri ve **`Depth Anything V2 Small (ViT-S)` (Apache-2.0)** yüzey normali sinyalleriyle damıtılmakta, tek geçişli sıcaklık kalibrasyonlu **Serbest Enerji (Free Energy) OOD** skoruyla bilinmeyen çatı engellerini saptamaktadır.

İki temel mühendislik kısıtı tasarım aşamasında kilitlenmiştir:
1. **Yerel Donanım Kısıtı (`GTX 1650` / `Turing TU117`):** Tensor Çekirdeği bulunmayan bu kartta FP16 yürütme gecikme gerilemesine yol açabildiğinden, yerel WSL2 duman testinde `ORT-CUDA FP32` (~238.0 MB toplam VRAM) ve `FP16` (~189.0 MB) profilleri yan yana ölçülerek en hızlı profil seçilir.
2. **Monoküler Eğim Kısıtı (Mod B):** 20.000 denemelik hata yayılımı ve 2026 hava derinlik literatürü (`AerialMetric` ECCV 2026, ISPRS XLIX-B4 2026), Mod B monoküler derinliğin 80–130 m irtifalarda 6.77–18.68 derece MAE eğim hatasına ulaştığını gösterdiğinden, 30 m üzeri irtifalarda Mod B ile `COMMIT_TOUCHDOWN` donanım/politika kilidiyle yasaklanır.

---

## Bölüm A — Doğrulanmış Saf İzinli Veri Seti Manifestosu ve Kanonik 5 Sınıflı SLZ Taksonomisi

### A.1. 2025–2026 Lisans Denetimi ve Gizli NC/ND/ShareAlike Tuzaklarının Tasfiyesi

Apache-2.0 lisanslı ağırlıkların ticari havacılık ve savunma sistemlerinde hukuki risk olmadan dağıtılabilmesi için kod depolarının yanında ham veri setlerinin birincil lisans hükümleri denetlenmiştir:

- **`OpenEarthMap` (41 / 100 Girdi Beyaz Liste):** Zenodo dağıtım kaydına göre etiketler kaynak görüntünün lisansını miras alır; ancak kaynak görüntünün Kamu Malı olduğu veya lisansının belirtilmediği bölgelerde etiketler otomatik olarak `CC BY-NC-SA 4.0` altına düşer [14]. Resmi atıf listesindeki 100 kaynak girdisinin 35'i `CC BY 4.0`, 6'sı `DL-DE-BY-2.0` (GeoNRW: Aachen, Bielefeld, Dortmund, Düsseldorf, Köln, Münster) olmak üzere toplam **41 girdi (%41) izinlidir**; kalan **59 girdi (%59)** (36 `CC BY-NC-SA 4.0` Landcover.ai/xBD, 12 belirsiz Kyoto/Tokyo/Peru, 6 `CC BY-SA 4.0` SpaceNet, 5 Kamu Malı Inria) veri yükleyicide kesin olarak dışlanır.
- **`FloodNet-Supervised v1.0` vs `RescueNet`:** Resmi `CDLA-Permissive-1.0` lisansını taşıyan veri seti RescueNet değil **`FloodNet-Supervised_v1.0`** sürümüdür (`FloodNet-VQA` sürümü `CC BY-NC-ND` olduğu için dışlanır) [14]. **`RescueNet`** makalesi `CC BY 4.0` olsa da BinaLab'ın resmi GitHub `README.md` dosyası 4.494 görüntülük veri setini **`CC BY-NC-ND 4.0`** olarak lisanslamaktadır; bu nedenle RescueNet kesin olarak elenmiştir.
- **Diğer Reddedilen Adaylar:** **`LoveDA`** (yalnızca akademik kullanım ve Google Earth ToS), **`DroneDeploy`** (`dd-ml-segmentation-benchmark`, lisans metni yok), **`FlyAwareV2`** (`GPL-3.0` + yasaklı `UAVid`) ve **`DINOv3`** (`LICENSE.md` ITAR/askerî/savunma/nükleer/tersine mühendislik yasakları) [10, 11] eğitimden çıkarılmıştır. **`SynDrone`** (`Apache-2.0` kod; ayrı veri lisansı ve sınıf ID belgesi eksik) yerine açık **`MIT`** lisanslı **`SkyScenes` (ECCV 2024)** [2], gerçek kentsel çatı + DSM yükseklik denetimi içinse Fransız IGNF'nin `CC-BY`/`OGL` uyumlu **`FLAIR-1` (`OPEN LICENCE 2.0 / etalab-2.0`)** veri seti [14] manifestoya alınmıştır.

#### OpenEarthMap 100 Kaynak Girdisi Lisans Kırılımı

| Lisans Kategorisi | Girdi Sayısı | Pay (%) | Eğitim Durumu | Kapsam & Örnek Bölgeler |
| :--- | :---: | :---: | :--- | :--- |
| `CC BY 4.0` | 35 | 35 | **İzinli (Beyaz Liste)** | Christchurch (AIRS), Kişinev (HTCD), Open Cities AI Afrika kentleri, Melbourne, Buenos Aires, Santiago, Bogota, Dakka, Ulan Batur, Maputo, Chiangmai, Baybay, Al Qurnah vb. |
| `DL-DE-BY-2.0` | 6 | 6 | **İzinli (Beyaz Liste)** | GeoNRW (Almanya): Aachen, Bielefeld, Dortmund, Düsseldorf, Köln, Münster. |
| `CC BY-NC-SA 4.0` | 36 | 36 | **Reddedildi (NC + SA)** | Polonya Landcover.ai (14 bölge) ve Maxar xBD (22 bölge: Adelaide, Houston, Mexico City, Palu vb.). |
| Belirsiz -> `CC BY-NC-SA 4.0` | 12 | 12 | **Reddedildi (Otomatik NC-SA)** | Kyoto, Tokyo (Japonya) ve 10 Peru bölgesi (Lima, Piura, Ica vb.). |
| `CC BY-SA 4.0` | 6 | 6 | **Reddedildi (ShareAlike)** | Maxar SpaceNet (6 kent): Rio, Şanghay, Paris, Rotterdam, Hartum, Vegas. |
| Kamu Malı -> `CC BY-NC-SA 4.0` | 5 | 5 | **Reddedildi (Otomatik NC-SA)** | Inria (5 bölge): Tyrol, Viyana, Austin, Chicago, Kitsap. |

#### Tam Lisans Denetim Manifestosu (Eğitim, Eval-Only ve Yasaklı Bileşenler)

| Bileşen / Veri Seti | Doğrulanmış Lisans (Kod & Veri) | Durum & Rol | Çözünürlük / Ölçek / Modalite | Doğrudan Kaynak & Denetim Notu |
| :--- | :--- | :--- | :--- | :--- |
| `DINOv2-ViT-L/14` & `ViT-B/14` | **Apache-2.0** (Kod + Ağırlık) | **ONAYLI (Öğretmen 1)** | 1024-boyut yoğun yama gömümü (`518x518`) | `github.com/facebookresearch/dinov2` — Ticari ve savunma türevlerine tamamen açık. |
| `Depth Anything V2 Small (ViT-S)` | **Apache-2.0** (Kod + ViT-S Göreli & Metrik VKITTI) | **ONAYLI (Öğretmen 2 & Mod B)** | 24.8M param, `518x518` göreli derinlik | `huggingface.co/depth-anything/Depth-Anything-V2-Small` — ViT-B/L/G (`CC-BY-NC-4.0`) kesin yasaklı. |
| `PIDNet-S` | **MIT** | **ONAYLI (Kazanan Öğrenci)** | 7.6M taban (7.75M +aux), `512x512` | `github.com/XuJiacong/PIDNet` — Temiz izinli omurga eğitimi. |
| `FLAIR-1` (IGN Fransa) | **OPEN LICENCE 2.0 (`etalab-2.0`)** (`CC-BY` / `OGL` uyumlu) | **ONAYLI (Eğitim — Gerçek)** | >20 milyar piksel, `512x512`, 0.2 m GSD, RGB+NIR+DSM | `huggingface.co/datasets/IGNF/FLAIR-1-2` — Gerçek çatı + DSM yükseklik denetimi. |
| `OpenEarthMap` (Filtreli 41 Girdi) | **`CC BY 4.0`** (35) + **`DL-DE-BY-2.0`** (6) | **ONAYLI (Eğitim — Gerçek, Beyaz Liste)** | 5.000 hava/uydu karesi (tüm set 97 bölge), 8 sınıf | `zenodo.org/records/7223446` — 59 NC-SA/SA/Kamu Malı/Belirsiz girdi filtrelenir. |
| `FloodNet-Supervised v1.0` | **`CDLA-Permissive-1.0`** | **ONAYLI (Eğitim — Gerçek İHA)** | 2.343 kare (`4000x3000`), alçak irtifa İHA, 10 sınıf | `github.com/BinaLab/FloodNet-Supervised_v1.0` — `FloodNet-VQA` (`CC BY-NC-ND`) kullanılmaz. |
| `SkyScenes` (ECCV 2024) | **`MIT`** | **ONAYLI (Eğitim — Dış Sentetik)** | 33.600 kare, 8 CARLA kenti, 15/35/60 m, 0–90 derece, RGB-D + 28 sınıf | `huggingface.co/datasets/hoffman-lab/SkyScenes` — `SynDrone`/`FlyAwareV2` yerine birincil dış sentetik kaynak. |
| Gazebo-İstanbul 8K (Yazar Üretimi) | **Apache-2.0** (Yazar mülkiyeti) + **Copernicus GLO-30 Açık Lisans** | **ONAYLI (Eğitim — Yerel Sentetik)** | 8.000 kare (`512x512`), 15–120 m, 0–35 derece, RGB-D+Normal+OOD | Colab-G4 EGL üretimi; OSM `ODbL` ayak izleri ağırlık eğitiminden tamamen izole edilir. |
| `SynDrone` (ICCVW 2023) | Kod: `Apache-2.0`; ayrı veri lisansı ve sınıf tablosu belgelenmemiş | **İHTİYATİ / KULLANILMAZ** | 8 CARLA kenti, RGB+Derinlik+LiDAR | `lttm.dei.unipd.it/paper_data/syndrone/` — `SkyScenes (MIT)` tam lisans açıklığıyla ikame eder. |
| `RescueNet` | Kod: `MIT`; **Veri: `CC BY-NC-ND 4.0`** | **REDDEDİLDİ (Eğitim Yasak)** | 4.494 İHA karesi (`3000x4000`), 11 sınıf | Resmi BinaLab deposu `NC-ND`'dir; üçüncü parti HF `CC-BY-4.0` etiketi yanıltıcıdır. |
| `LoveDA` | **Yalnızca Akademik / Google Earth ToS** (`CC BY-NC-SA 4.0`) | **REDDEDİLDİ (Eğitim Yasak)** | 5.987 görüntü (`1024x1024`), 0.3 m GSD, 7 sınıf | Ticari kullanım kesinlikle yasaktır. |
| `DroneDeploy` | **Lisans Belirtilmemiş** (Varsayılan tüm hakları saklı) | **REDDEDİLDİ (Eğitim Yasak)** | 6 sınıf + Ignore ortofoto | `github.com/dronedeploy/dd-ml-segmentation-benchmark` açık lisans içermez. |
| `FlyAwareV2` (2026) | **`GPL-3.0`** + `UAVid` (`CC BY-NC-SA 4.0`) | **REDDEDİLDİ (Eğitim Yasak)** | >100K sentetik+gerçek kare | Copyleft kod ve yasaklı `UAVid` içeriği nedeniyle elendi. |
| `DINOv3` | **Özel DINOv3 Lisansı** (ITAR/Askerî/Savunma/Nükleer yasaklı) | **KESİN YASAKLI** | ViT-S/16 – ViT-7B/16 | `LICENSE.md` ITAR/askerî yasaklar ve zorunlu lisans devri içerir. |
| `Mid-Air` & `UAVid` | **`CC BY-NC-SA 4.0`** | **KESİN YASAKLI** | Kırsal orman / kentsel İHA | NC + ShareAlike nedeniyle tamamen yasaklı. |
| `AeroScapes` | **`CC BY-SA 4.0`** (ShareAlike) | **SADECE EVAL-ONLY (Sıfır-Atış)** | 3.269 kare (`1280x720`), 5–30 m, 12 sınıf | Ağırlıklara asla girmez; gerçek dünya sıfır-atış değerlendirme. |
| `ICG Semantic Drone` | **Ticari Olmayan (NC)** | **SADECE EVAL-ONLY (Sıfır-Atış)** | 600 (400 açık) kare (`6000x4000`), 5–50 m, 24 sınıf | Yüksek çözünürlüklü kentsel çatı/avlu sıfır-atış testi. |
| `UAVScenes` (ICCV 2025) | **`CC BY-NC-SA 4.0`** | **SADECE EVAL-ONLY (RGB+LiDAR)** | 120.000 kare (`2448x2048`) + Livox Avia LiDAR, 80/90/130 m, 19 aktif sınıf | `github.com/sijieaaa/UAVScenes` — Mod B monoküler derinlik/eğim ablasyonu yer gerçeği. |

**Coğrafi Veri (Copernicus DEM GLO-30 ve OSM `ODbL`) Sıfır-ShareAlike Güvencesi:** Maltepe–Dragos–Kartal simülasyon dünyasının arazi yükselti katmanında kullanılan **Copernicus DEM GLO-30** (`Copernicus_DSM_COG_10_N40_00_E029_00_DEM`), AB/ESA/DLR/Airbus tarafından ticari kullanım, uyarlama ve yeniden dağıtıma tam izin veren **Tam, Ücretsiz ve Açık Lisans** ile sunulur (yalnızca `NOTICE` atfı gerektirir) [15]. Buna karşılık **OpenStreetMap (OSM)** ve Overture Maps bina katmanları **`ODbL 1.0`** altındadır ve OSMF *Attribution Guidelines Bölüm 1.5.6* aşırı eğitilmiş makine öğrenmesi modellerinin Türev Veritabanı (Derivative Database) sayılma riskine dikkat çeker. Bu nedenle eğitim ağırlıklarına giren 8.000 karelik Gazebo İstanbul veri seti **Copernicus GLO-30 kıyı topoğrafyası üzerine İstanbul kıyı mimari dağılımlarıyla prosedürel üretilen rastgele binalar, teras çatılar ve sahil parklarıyla** kurulur; OSM bina ayak izleri ağırlık eğitiminden tamamen izole edilen ayrı bir görselleştirme/test dünyasında tutulur.

### A.2. Kanonik 5 Sınıflı SLZ Semantik Taksonomisi ve Eşleme JSON'u

| Kanonik ID | Sınıf Adı | Kapsam & Operasyonel Güvenlik Kuralı (`VertiRTA-X8-240`) |
| :---: | :--- | :--- |
| `0` | `Safe_Flat_Ground` | Asfalt yol, kaldırım, otopark, pist, çıplak/sıkıştırılmış düz toprak ve engelsiz düz zemin (3B eğim <= 8 derece ve pürüzlülük <= 0.15 m doğrulamasıyla aday SLZ). |
| `1` | `Safe_Flat_Roof` | Düz veya teras bina çatıları. `OpenEarthMap`, `FLAIR-1`, `FloodNet` ve `SkyScenes` içinde `building` etiketi eğimli çatı ve duvarları da kapsayabildiğinden, eğimli yüzeylerin elenmesi Bölüm C'deki 3B düzlem geçidiyle garanti edilir; sel basmış/hasarlı binalar (`FloodNet`) ve güneş panelli/şeffaf çatılar (`UAVScenes`) doğrudan `2` sınıfına atanır. |
| `2` | `Hazard_Obstacle_Structure` | Düşey bina cepheleri, parapet duvarları, çitler, direkler, konteynerler, köprü yapıları, klima/HVAC üniteleri, güneş panelleri, seralar, kayalıklar ve hasarlı yapılar. |
| `3` | `Hazard_Dynamic_Human_Vehicle` | Yayalar, bisiklet/motosiklet sürücüleri, otomobiller, kamyonlar, otobüsler, hayvanlar, tekneler ve diğer İHA'lar (2.5B ızgarada 2x tehlike ağırlığı ve yavaş üstel sönümleme). |
| `4` | `Hazard_Water_Vegetation` | Deniz/nehir/göl, yüzme havuzları, ağaçlar, çalılıklar, orman, tarım/bağ ve otlaklar. 240 kg iniş takımı batma ve gizli çukur riskine karşı kısa çim (`grass`/`green_field`) de muhafazakâr biçimde `4` altında tutulur. |
| `254` | `OOD_Synthetic_Outlier` | Yalnızca eğitimde Gazebo-İstanbul ve AnomalyMix'ten gelen, çapraz entropide yoksayılıp enerji marjini kaybıyla yüksek serbest enerjiye itilen bilinmeyen çatı engelleri (branda, iskele, parıltı, şişme havuz). |
| `255` | `Ignore` | Gökyüzü, verisiz kenar pikselleri, etiketlenmemiş/çelişkili pikseller ve `UAVScenes` boş ID'leri. |

> **Eğik Görünüm Cephe Filtresi (`SkyScenes`):** Eğik açılarda (45 derece, 60 derece) `SkyScenes` içindeki `building` (ID `1`) pikselleri hem düz çatıları hem dikey cephe duvarlarını kapsar. Veri yükleyici, her karenin derinlik haritası ve bilinen kamera eğimi üzerinden dünya-koordinatı yüzey normalini hesaplar; $|n_z| < 0.85$ (~31.8 derece üzeri cephe/duvar) koşulunu sağlayan pikselleri otomatik olarak **`2: Hazard_Obstacle_Structure`** sınıfına çevirir.

```json
{
  "schema_version": "vertirta_slz_taxonomy_v1.0",
  "canonical_classes": {
    "0": "Safe_Flat_Ground",
    "1": "Safe_Flat_Roof",
    "2": "Hazard_Obstacle_Structure",
    "3": "Hazard_Dynamic_Human_Vehicle",
    "4": "Hazard_Water_Vegetation",
    "254": "OOD_Synthetic_Outlier",
    "255": "Ignore"
  },
  "train_datasets_permissive_only": {
    "FLAIR_1_etalab2": {
      "license": "OPEN LICENCE 2.0 (etalab-2.0)",
      "index_base": 1,
      "mapping": {
        "1": 1, "2": 0, "3": 0, "4": 0,
        "5": 4, "6": 4, "7": 4, "8": 4, "9": 4, "10": 4, "11": 4, "12": 4, "13": 4,
        "14": 255, "15": 4, "16": 4, "17": 4, "18": 2, "19": 255
      },
      "notes": "1:building->1 (3B geometriyle filtrelenir); 2:pervious,3:impervious,4:bare_soil->0; 5..13,15..17:water/veg/pool/plowed->4; 18:greenhouse->2; 14:snow,19:other->255."
    },
    "OpenEarthMap_filtered_41_entries": {
      "license": "CC-BY-4.0 (35) + DL-DE-BY-2.0 (6) ONLY",
      "raw_pixel_values_0_to_8": {
        "0": 255, "1": 0, "2": 4, "3": 0, "4": 0, "5": 4, "6": 4, "7": 4, "8": 1
      },
      "zero_indexed_loader_0_to_7": {
        "0": 0, "1": 4, "2": 0, "3": 0, "4": 4, "5": 4, "6": 4, "7": 1
      },
      "notes": "Raw 0:No-Data->255; 1:Bareland,3:Developed_space,4:Road->0; 2:Rangeland,5:Tree,6:Water,7:Agriculture->4; 8:Building->1. Yalnızca 41 izinli girdi yüklenir."
    },
    "FloodNet_Supervised_v1_0": {
      "license": "CDLA-Permissive-1.0",
      "index_base": 0,
      "mapping": {
        "0": 255, "1": 2, "2": 1, "3": 4, "4": 0,
        "5": 4, "6": 4, "7": 3, "8": 4, "9": 4
      },
      "notes": "0:Background->255; 1:Building-flooded->2; 2:Building-non-flooded->1; 3:Road-flooded,5:Water,6:Tree,8:Pool,9:Grass->4; 4:Road-non-flooded->0; 7:Vehicle->3."
    },
    "SkyScenes_MIT": {
      "license": "MIT",
      "index_base": 0,
      "mapping": {
        "0": 255, "1": 1, "2": 2, "3": 255, "4": 3, "5": 2,
        "6": 0, "7": 0, "8": 0, "9": 4, "10": 3, "11": 2,
        "12": 2, "13": 255, "14": 0, "15": 2, "16": 2, "17": 2,
        "18": 2, "19": 2, "20": 3, "21": 4, "22": 4, "23": 3,
        "24": 3, "25": 3, "26": 3, "27": 3
      },
      "notes": "1:building->1 (|n_z|<0.85 cephe pikselleri veri yükleyicide 2'ye çevrilir); 4,10,20,23..27->3; 6,7,8,14->0; 9,21,22->4; 2,5,11,12,15..19->2; 0,3,13->255."
    },
    "Gazebo_Istanbul_8K_Author": {
      "license": "Apache-2.0 (Author-Owned)",
      "sdf_label_ids": {
        "10": 0, "20": 1, "30": 2, "40": 3, "50": 4, "254": 254, "0": 255
      },
      "notes": "10:Asfalt/Kıyı-Zemin/Helipad; 20:Düz-Teras-Çatı; 30:Parapet/HVAC/Eğimli-Çatı/Cephe/Direk; 40:Yaya/Araç; 50:Marmara-Denizi/Ağaç/Çim; 254:OOD-Branda/İskele/Parıltı."
    }
  },
  "eval_only_datasets_never_trained": {
    "AeroScapes_CC_BY_SA_4": {
      "license": "CC-BY-SA-4.0 (EVAL-ONLY)",
      "index_base": 0,
      "mapping": {
        "0": 255, "1": 3, "2": 3, "3": 3, "4": 3, "5": 3,
        "6": 3, "7": 2, "8": 2, "9": 4, "10": 0, "11": 255
      },
      "notes": "0:Background,11:Sky->255; 1..6(Person,Bike,Car,Drone,Boat,Animal)->3; 7:Obstacle,8:Construction->2; 9:Vegetation->4; 10:Road->0."
    },
    "ICG_Semantic_Drone_NC": {
      "license": "Non-Commercial (EVAL-ONLY)",
      "index_base": 0,
      "mapping": {
        "0": 255, "1": 0, "2": 0, "3": 4, "4": 0, "5": 4,
        "6": 2, "7": 4, "8": 4, "9": 1, "10": 2, "11": 2,
        "12": 2, "13": 2, "14": 2, "15": 3, "16": 3, "17": 3,
        "18": 3, "19": 4, "20": 4, "21": 0, "22": 2, "23": 255
      },
      "notes": "0-indeksli CSV eşlemesi (1-indeksli maskelerde ID-1 uygulanır). 1:paved,2:dirt,4:gravel,21:ar-marker->0; 9:roof->1; 6,10..14,22->2; 15..18->3; 3,5,7,8,19,20->4; 0,23->255."
    },
    "UAVScenes_CC_BY_NC_SA_4": {
      "license": "CC-BY-NC-SA-4.0 (EVAL-ONLY, RGB+LiDAR)",
      "index_base": 0,
      "mapping": {
        "0": 255, "1": 1, "2": 0, "3": 0, "4": 4, "5": 4,
        "6": 2, "7": 255, "8": 255, "9": 2, "10": 0, "11": 2,
        "12": 255, "13": 4, "14": 4, "15": 2, "16": 2, "17": 2,
        "18": 0, "19": 0, "20": 3, "21": 255, "22": 255, "23": 255,
        "24": 3, "25": 255
      },
      "notes": "cmap.py seyrek ID'leri: 1:roof->1; 2,3,10,18,19->0; 6,9,11,15,16,17->2; 20,24->3; 4,5,13,14->4; boş ID'ler (7,8,12,21..23,25)->255."
    }
  }
}
```

---

## Bölüm B — Colab G4 96 GB Temel Modelden Gerçek Zamanlı Öğrenciye Damıtma ve C++20 ONNX/TRT Dışa Aktarımı

### B.1. Öğrenci Mimarisi Karşılaştırması (`512x512`) ve Tek Kazanan Seçimi

Hava SLZ segmentasyonunda çatı parapetleri, HVAC üniteleri ve küçük çatı engellerinin keskin sınırlarla ayrıştırılması; aynı zamanda `GTX 1650` (4 GB) üzerinde 300 MB altı FP16 VRAM ve tek alt-graf TensorRT 10 uyumluluğu temel ölçüttür. Üç aday mimari `512x512` giriş ve 3 kanallı yardımcı yüzey normali başlığı (`+0.15M` param, `+0.25 GFLOPs`) ile karşılaştırılmıştır:

| Mimari (`512x512` Statik) | Lisans | Parametre | GFLOPs (`512x512`) | FP16 Ağırlık / Aktivasyon+I/O / Toplam VRAM (140 MB CUDA Bağlamlı) | Yayımlı Test mIoU (Cityscapes / CamVid) | Karar |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`PIDNet-S` (+aux normal)** | **MIT** | **7.75M** (7.6M + 0.15M) | **6.20** | **15.5 MB / 33.5 MB / ~189.0 MB** (-%37 pay) | **%78.6 / %80.1** (+1.2 / +1.5 puan) | **KAZANAN:** Açık Türev (D) sınır dalı + Bag Fusion + statik PAPPM ile tek alt-graf TRT. |
| `DDRNet-23-slim` (+aux normal) | MIT | 5.85M (5.7M + 0.15M) | 4.79 | 11.7 MB / 27.5 MB / ~179.2 MB | %77.4 / %78.6 | **ELENDİ:** 1.41 GFLOPs daha hafif olsa da açık sınır (D) dalı yok; parapet sınır doğruluğu ve mIoU daha düşük, `align_corners=False` duyarlı. |
| `SegFormer-B0` (MiT-B0) | **NVIDIA Source Code (NC!)** | 3.80M | 8.40 | 7.6 MB / 49.5 MB / ~197.1 MB | Lisans nedeniyle elendi | **REDDEDİLDİ:** Resmi ağırlıklar ticari olmayan (NC) lisanslı; en yüksek GFLOPs ve en yüksek aktivasyon belleği (44.0 MB). |

**Seçim Gerekçesi (`PIDNet-S`):** `PIDNet-S`, Oransal (P — yüksek çözünürlüklü detay), İntegral (I — küresel bağlam) ve Türev (D — yüksek frekanslı sınır tespiti) dallarını Bag Fusion modülüyle birleştirir [12]. D dalından çıkan sınır olasılık haritası $b(u) \in [0, 1]$ yalnızca çatı/parapet sınırlarını keskinleştirmekle kalmaz, Bölüm C'deki Serbest Enerji OOD skorunda sınıf-sınırı yanlış alarmlarını doğrudan telafi etmek için de yeniden kullanılır. ONNX Opset 17+ / TensorRT FP16 uyumluluğu için PAPPM içindeki dinamik `AdaptiveAvgPool2d` katmanları, statik `1x3x512x512` girişe kilitlenmiş sabit çekirdek/adımlı `AvgPool2d` ve küresel ortalama (`mean(dim=(2,3), keepdim=True)`) ile değiştirilerek veri-bağımlı şekil (DDS) bölünmesi sıfırlanır [15].

### B.2. Colab G4 (96 GB VRAM) Temel Modelden Öğrenciye Damıtma Hattı

Colab G4 (`NVIDIA RTX PRO 6000 Blackwell`, `sm_120`, 96 GB VRAM) üzerinde dondurulmuş iki **Apache-2.0** temel öğretmen (`DINOv2-ViT-L/14` + `Depth Anything V2 Small`) BF16 hassasiyetinde yalnızca ~0.7 GB VRAM kaplayarak çevrimiçi (on-the-fly) denetim sağlar ve büyük yığın boyutlarıyla (`batch_size = 32`) kararlı damıtma yürütülür:

1. **Dondurulmuş `DINOv2-ViT-L/14` Yoğun Yama Damıtması ($\mathcal{L}_{\text{feat\_dino}}$):** Giriş RGB karesi `518x518` boyutuna (yama boyutu 14'ün katı -> `37x37` yoğun yama belirteci, 1024 kanal) ölçeklenir. Öğrencinin Bag Fusion sonrası sondan bir önceki öznitelik haritası, yalnızca eğitimde kullanılan hafif bir projektör (`1x1 Conv -> GroupNorm -> GELU -> 1x1 Conv`) ile 1024 kanala yansıtılır, `37x37` uzamsal boyutuna çift doğrusal hizalanır ve yama bazlı kosinüs benzerliği kaybıyla eğitilir [11]:
$$\mathcal{L}_{\text{feat\_dino}} = 1 - \frac{1}{|\mathcal{P}|}\sum_{p \in \mathcal{P}} \frac{\hat{\mathbf{z}}_p \cdot \mathbf{z}_p^{\text{DINO}}}{\|\hat{\mathbf{z}}_p\|_2\,\|\mathbf{z}_p^{\text{DINO}}\|_2}$$
   *(Projektör başlığı ONNX dışa aktarımında atılır; çıkarımda 0 FLOPs / 0 VRAM ek yük.)*
2. **Dondurulmuş `Depth Anything V2 Small` Geometrik Normal Damıtması ($\mathcal{L}_{\text{geom\_dav2}}$):** Metrik derinlik veya DSM yer gerçeği içeren veri setlerinde (`FLAIR-1`, `SkyScenes`, `Gazebo-İstanbul 8K`) yardımcı 3 kanallı yüzey normali başlığı $\hat{\mathbf{n}}(u)$ doğrudan yer gerçeği normalleriyle eğitilir. Derinlik yer gerçeği olmayan gerçek RGB karelerde (`OpenEarthMap`, `FloodNet`), dondurulmuş `Depth Anything V2 Small` göreli ters derinlik haritasından sonlu farklarla türetilen sözde birim yüzey normalleri $\mathbf{n}^{\text{DAv2}}(u)$ kullanılır (Depth Anything V2 protokolü izlenerek gökyüzü ve en yüksek kayıplı %10 gürültülü piksel maskelenir [1]):
$$\mathcal{L}_{\text{geom\_dav2}} = \frac{1}{|\Omega_{\text{geo}}|} \sum_{u \in \Omega_{\text{geo}}} \left[\left(1 - \hat{\mathbf{n}}(u)\cdot \mathbf{n}^*(u)\right) + 0.5\,\|\hat{\mathbf{n}}(u) - \mathbf{n}^*(u)\|_1\right]$$
3. **Bileşik Kayıp Fonksiyonu:**
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{ce\_lovasz}} + \lambda_{\text{bdry}}\,\mathcal{L}_{\text{bdry}} + \lambda_{\text{dino}}\,\mathcal{L}_{\text{feat\_dino}} + \lambda_{\text{geom}}\,\mathcal{L}_{\text{geom\_dav2}} + \lambda_{\text{ood}}\,\mathcal{L}_{\text{energy\_margin}}$$
   Burada $\mathcal{L}_{\text{ce\_lovasz}}$, Yanlış-Güvenli Oranını (False-Safe Rate) bastırmak üzere tehlike sınıflarına (`2` ve `3`) 1.5x ağırlık veren sınıf-ağırlıklı çapraz entropi ile Lovasz-Softmax toplamıdır; $\mathcal{L}_{\text{bdry}}$ ise PIDNet D-dalı için ağırlıklı ikili çapraz entropi sınır kaybıdır.

#### Eğitim Hiperparametreleri ve Sim-to-Real Çöküşünü Önleyen Yığın Örnekleme Oranı

| Parametre / Aşama | Aşama 1: Temel Damıtma & Segmentasyon | Aşama 2: OOD Enerji Marjini İnce Ayarı |
| :--- | :--- | :--- |
| Donanım & Hassasiyet | Colab G4 (`RTX PRO 6000 Blackwell`, 96 GB), BF16 AMP | Colab G4, BF16 AMP (omurga düşük LR veya son başlık odaklı) |
| Epoch & Optimizer | 80 epoch, AdamW (`weight_decay = 1e-2`), `batch_size = 32` | 20 epoch, AdamW (`lr = 1e-5`, polinom azalım 0.9), `batch_size = 32` |
| Öğrenme Oranı (LR) | `3e-4` -> `1e-6` (5 epoch ısınma + kosinüs azalım) | `1e-5` -> `1e-6` |
| Kayıp Ağırlıkları | `lambda_bdry = 0.4`, `lambda_dino = 0.5`, `lambda_geom = 0.3`, `lambda_ood = 0.0` | `lambda_bdry = 0.4`, `lambda_dino = 0.2`, `lambda_geom = 0.2`, `lambda_ood = 0.1` |
| Enerji Marjini & OE | Kapalı | `m_in = -23.0`, `m_out = -5.0` (doğrulama logit ölçeğine kalibreli), Gazebo `label=254` + %10 olasılıklı AnomalyMix |
| **Gerçek / Sentetik Yığın Oranı** | **%60 Gerçek İzinli** (`FLAIR-1` %25, `OpenEarthMap-41` %20, `FloodNet` %15) / **%40 Sentetik** (`SkyScenes` %20, `Gazebo-İstanbul 8K` %20) | %55 Gerçek (+AnomalyMix) / %45 Sentetik (`Gazebo-İstanbul` OOD ağırlıklı %25 + `SkyScenes` %20) |
| Veri Artırımı (Augmentation) | Rastgele ölçek (`0.5–2.0x`), `512x512` kırpma, 90 derece döndürme/aynalama, renk/gölge/kıyı sisi fotometrik bozulması, hareket bulanıklığı | Aynı geometrik artırım + speküler parıltı ve OOD yama enjeksiyonu |

### B.3. PyTorch -> ONNX Opset 17 -> C++20 ONNX Runtime (`CUDA` / `TRT FP16`) Boru Hattı

```python
# export_pidnet_slz.py — Statik 1x3x512x512 ONNX Opset 17 Dışa Aktarımı
import torch, onnx, onnxsim

deploy_model = PIDNetSlzDeployWrapper(trained_model).eval().cuda()
dummy = torch.randn(1, 3, 512, 512, device="cuda", dtype=torch.float32)

torch.onnx.export(
    deploy_model, dummy, "vertirta_pidnet_s_512.onnx",
    opset_version=17, do_constant_folding=True,
    input_names=["rgb_nchw"],
    output_names=["seg_logits", "surf_normals", "bdry_prob"],
    dynamic_axes=None  # Kesin statik şekil -> sıfır DDS, tam CUDA Graph & TRT füzyonu
)
onnx_model, ok = onnxsim.simplify(onnx.load("vertirta_pidnet_s_512.onnx"))
assert ok
onnx.save(onnx_model, "vertirta_pidnet_s_512_sim.onnx")
```

C++20 çalışma zamanında ONNX Runtime 1.21–1.26 (`CUDA 12.8` + `cuDNN 9.x`) veya 1.27+ (`CUDA 13.0` / `CUDA 12.8` + `cuDNN 9.x`) sürümleri kullanılır [15]. Model topolojisi, ONNX Runtime sürümü veya TensorRT sürümü değiştiğinde `.engine` ve `.profile` önbellek dosyaları temizlenmeli; CUDA Graph etkinken `Session::Run()` yalnızca tek bir algılama iş parçacığından, aynı GPU bellek adresleri yerinde (`update_inplace`) güncellenerek çağrılmalıdır.

```cpp
// slz_ort_engine.hpp (C++20, SPDX-License-Identifier: Apache-2.0)
#include <onnxruntime_cxx_api.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <vector>

struct SlzOrtConfig {
  std::string onnx_path;
  std::string trt_cache_dir = "/content/drive/MyDrive/vertirta_cache/trt_sm120";
  bool use_tensorrt = true;       // Colab G4 Blackwell / Jetson Orin -> true
  bool enable_fp16 = true;        // GTX 1650 (TU117, Tensor Çekirdeksiz) -> FP32 ile kıyasla!
  bool enable_cuda_graph = true;  // Tek iş parçacıklı deterministik yürütme için
};

class SlzOrtRunner {
 public:
  explicit SlzOrtRunner(const SlzOrtConfig& cfg)
      : env_(ORT_LOGGING_LEVEL_WARNING, "VertiRTA_SLZ") {
    const auto& api = Ort::GetApi();
    Ort::SessionOptions opts;
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    opts.SetIntraOpNumThreads(1);
    cudaStreamCreate(&stream_);

    if (cfg.use_tensorrt) {
      OrtTensorRTProviderOptionsV2* trt_opts = nullptr;
      Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&trt_opts));
      std::unique_ptr<OrtTensorRTProviderOptionsV2, decltype(api.ReleaseTensorRTProviderOptions)>
          guard(trt_opts, api.ReleaseTensorRTProviderOptions);

      std::vector<const char*> keys = {
          "device_id", "trt_max_workspace_size", "trt_fp16_enable",
          "trt_engine_cache_enable", "trt_engine_cache_path",
          "trt_timing_cache_enable", "trt_timing_cache_path",
          "trt_builder_optimization_level", "trt_cuda_graph_enable"};
      std::vector<const char*> vals = {
          "0", "1073741824", cfg.enable_fp16 ? "1" : "0",
          "1", cfg.trt_cache_dir.c_str(),
          "1", cfg.trt_cache_dir.c_str(),
          "3", cfg.enable_cuda_graph ? "1" : "0"};
      Ort::ThrowOnError(api.UpdateTensorRTProviderOptions(trt_opts, keys.data(), vals.data(), keys.size()));
      Ort::ThrowOnError(api.UpdateTensorRTProviderOptionsWithValue(trt_opts, "user_compute_stream", stream_));
      opts.AppendExecutionProvider_TensorRT_V2(*trt_opts);
    }

    OrtCUDAProviderOptionsV2* cuda_opts = nullptr;
    Ort::ThrowOnError(api.CreateCUDAProviderOptions(&cuda_opts));
    std::unique_ptr<OrtCUDAProviderOptionsV2, decltype(api.ReleaseCUDAProviderOptions)>
        cguard(cuda_opts, api.ReleaseCUDAProviderOptions);
    std::vector<const char*> ckeys = {"device_id", "cudnn_conv_algo_search", "enable_cuda_graph"};
    std::vector<const char*> cvals = {"0", "HEURISTIC", cfg.enable_cuda_graph ? "1" : "0"};
    Ort::ThrowOnError(api.UpdateCUDAProviderOptions(cuda_opts, ckeys.data(), cvals.data(), ckeys.size()));
    Ort::ThrowOnError(api.UpdateCUDAProviderOptionsWithValue(cuda_opts, "user_compute_stream", stream_));
    opts.AppendExecutionProvider_CUDA_V2(*cuda_opts);

    session_ = std::make_unique<Ort::Session>(env_, cfg.onnx_path.c_str(), opts);
    AllocateStaticIoBindings(); // cudaMalloc: 1x3x512x512 in; 1x5x512x512 + 1x3x512x512 + 1x1x64x64 out
  }

 private:
  void AllocateStaticIoBindings();
  Ort::Env env_;
  std::unique_ptr<Ort::Session> session_;
  cudaStream_t stream_{nullptr};
};
```

#### Donanım Etiketli VRAM ve Gecikme / FPS Profili

| Donanım Platformu | Yürütücü / Hassasiyet | Tahminî Toplam VRAM (`512x512`) | Gecikme / FPS Profili (Referans vs Hedef Bütçe) |
| :--- | :--- | :--- | :--- |
| Colab G4 — `NVIDIA RTX PRO 6000 Blackwell` (`sm_120`, 96 GB) | `TensorrtExecutionProvider` FP16 + CUDA Graph | ~189.0 MB | **Hedef Bütçe:** `< 2.0 ms` salt ağ (`> 500 FPS`), `> 120 Hz` tam boru hattı *(Kapı 5'te ölçülecek)*. |
| Yerel WSL2 — `NVIDIA GTX 1650` (`Turing TU117`, 4 GB, Tensor Çekirdeksiz) | `CUDAExecutionProvider` FP32 vs FP16 | ~238.0 MB (FP32) / ~189.0 MB (FP16) | **Hedef Bütçe:** 6–12 ms salt ağ (`80–160 FPS`), 25–30+ Hz tam `ISlzPriorProvider`; `TU117` FP16 gerileme testi zorunlu. |
| Uç Referans — `NVIDIA Jetson Orin Nano 8 GB / AGX Orin` | `TensorrtExecutionProvider` FP16 | ~189.0 MB | **Hedef Bütçe:** `< 6.0 ms` salt ağ (`> 150 FPS`). |
| Yayımlı Referanslar (`RTX 3090` / `RTX 3050`) | PyTorch FP32 / ORT-TRT | — | **Doğrulanmış Literatür:** `RTX 3090` PyTorch FP32 `93.2 FPS` (`1024x2048`) ve `153.7 FPS` (CamVid) [12]; `RTX 3050` ORT-TRT `33.52 FPS` (+%35.60 hızlanma). |

---

## Bölüm C — Sıcaklık Kalibrasyonlu Tek Geçişli Serbest Enerji OOD ve Çift Modlu 3B Geometrik Füzyon

### C.1. Tek Geçişli Serbest Enerji OOD Formülasyonu ve Kalibrasyon Protokolü

`VertiRTA`'da Monte Carlo Dropout ($O(N)$ gecikme) ve Yoğun Kanıtsal Derin Öğrenme (EDL; epistemik belirsizliği sadık temsil etmediği gösterildiği için [4]) yasaklanmıştır. Tek ileri geçişte $C = 5$ kanonik logit vektörü $\mathbf{f}(u) = [f_0(u), \dots, f_4(u)]^\top$ üzerinden **Helmholtz Serbest Enerji Skoru** [13] hesaplanır:

$$E(u) = -T \log \sum_{c=0}^{4} \exp\!\left(\frac{f_c(u)}{T}\right)$$

`MaxLogit` ($-\max_c f_c(u)$) yalnızca $T \to 0^+$ limitine karşılık gelirken, sonlu sıcaklıkta Serbest Enerji tüm 5 sınıfın logit yoğunluğunu pürüzsüz biçimde birleştirerek girdi yoğunluğu $-\log p(x)$ ile hizalanır (Liu et al. 2020'de CIFAR-10 WideResNet üzerinde Softmax'a kıyasla FPR@95'i %51.04'ten %33.01'e, enerji marjini ince ayarıyla %3.32'ye indirmiştir [13]). İki ID sınıf arasındaki 1–2 piksellik sınırların yüksek enerji üretmesi zaafı, `PIDNet-S`'nin Türev (D) sınır başlığı $b(u) \in [0, 1]$ ile telafi edilir:

$$\tilde{E}(u) = E(u) - \kappa_b\,b(u), \qquad \kappa_b = 1.5$$

Sıcaklık ve eşik kalibrasyonu iki adımlı protokolle sabitlenir:
1. **Aşama-2 Sentetik Aykırı Değer Maruziyeti (OE):** Gazebo-İstanbul 8K içindeki `label=254` OOD çatı nesneleri (branda, metal yansıtıcı, şişme havuz, iskele, parıltı) ve AnomalyMix yamaları $\Omega_{\text{out}}$ kümesini oluşturur:
$$\mathcal{L}_{\text{energy\_margin}} = \frac{1}{|\Omega_{\text{in}}|} \sum_{u \in \Omega_{\text{in}}} \max\!\left(0,\; E(u) - m_{\text{in}}\right)^2 + \frac{1}{|\Omega_{\text{out}}|} \sum_{u \in \Omega_{\text{out}}} \max\!\left(0,\; m_{\text{out}} - E(u)\right)^2$$
2. **Eğitim Sonrası Sabitleme:** Dondurulmuş ağırlıklar üzerinde temiz doğrulama kümesinde NLL en küçükleyen olasılık sıcaklığı $T_{\text{prob}}^*$ (tipik 1.15–1.45) ve OOD sıcaklığı $T_{\text{ood}} = 1.0$ sabitlenir; karar eşiği $\tau_{\text{ood}}$, temiz ID doğrulama piksellerinde %95 doğru kabul ($\text{TPR}_{\text{ID}} = 95\%$) sağlayan 95. yüzdelik $\tilde{E}(u)$ değerine ayarlanır.

### C.2. Saf C++20 Çift Modlu 3B Geometrik Düzlük ve Açıklık Boru Hattı

**Mod A — Birincil Canlı SITL / RGB-D Modu:** Geçerli her derinlik pikseli $(u, v, Z)$ ($Z \in [0.8, 150.0]$ m), kamera iç parametreleri (`512x512`, 80 derece HFOV için $f_x = f_y = 305.09$ px) ve sabit optik eksen dönüşümüyle dünya ENU koordinatına geri-izdüşürülür. Gazebo Harmonic `rgbd_camera` eklentisindeki `PointCloud2` eksen kaymasını (`gz-sensors#545` [15]) aşmak için nokta bulutu konusu yerine ham `depth_image` ve deterministik $R_{\text{opt}\to\text{cam}}$ matrisi kullanılır:

$$\mathbf{p}_C = Z \begin{bmatrix} (u - c_x)/f_x \\ (v - c_y)/f_y \\ 1 \end{bmatrix}, \qquad \mathbf{p}_W = R_{WC}\,R_{\text{opt}\to\text{cam}}\,\mathbf{p}_C + \mathbf{t}_{WC}$$

Her `0.5 m x 0.5 m` hücre artımlı birinci ve ikinci moment toplamlarını ($N$, $\mathbf{s} = \sum \mathbf{p}_W$, $M = \sum \mathbf{p}_W \mathbf{p}_W^\top$) biriktirir. Kamera GSD'si 80 m AGL'de 26.22 cm/px (~3.6 piksel/hücre), 22 m inceleme irtifasında ise 7.21 cm/px (~48.1 piksel/hücre) olur. `3x3` komşuluk penceresinin (`1.5 m x 1.5 m`, >= 12 nokta) kovaryans matrisi $\Sigma = \frac{1}{N_{\text{win}}}M_{\text{win}} - \bar{\mathbf{p}}\bar{\mathbf{p}}^\top$ üzerinde analitik 3x3 simetrik Jacobi/Cardano özdeğer ayrıştırması uygulanarak en küçük özdeğer $\lambda_0$ ve $n_z > 0$ özvektöründen yerel eğim $\theta = \arccos(\text{clamp}(n_z, -1, 1)) \le 8.0$ derece ve pürüzlülük $\sigma_z = \sqrt{\max(0, \lambda_0)} \le 0.15$ m elde edilir. 5.0 m yarıçaplı dairesel rotor açıklık diskindeki **317 hücre** (disk alanı 79.25 m^2, gerçek $\pi R^2 = 78.54$ m^2) için azami dikey sapma $\max_{c \in \mathcal{D}_R} |(\bar{\mathbf{p}}_c - \bar{\mathbf{p}}_{\text{disk}}) \cdot \mathbf{n}_{\text{disk}}| \le 0.25$ m kuralı çatı parapetlerini yakalar.

**Mod B — RGB-Only Monoküler Yedek / Ablasyon Modu:** **`Depth Anything V2 Small` (Apache-2.0)** ile Metric3D tarzı kanonik kamera normalizasyonu [3] **giriş uzayında yeniden örnekleme** ile uygulanır ($f_c = 1000$ px oranına göre ölçekleme/kırpma), ardından ağın ürettiği ters göreli derinlik $\hat{\rho}(u,v)$ görüntü merkezinin %20 çekirdek bölgesi ($\mathcal{C}$) üzerinden barometre veya lazer altimetre AGL değeriyle metrik ölçeğe çapalanır:

$$\hat{Z}_{\text{rel}}(u,v) = \frac{1}{\hat{\rho}(u,v) + \epsilon}, \qquad s = \frac{z_{\text{AGL}} / \cos(\phi_{\text{pitch}})}{\text{median}_{(u,v)\in\mathcal{C}}\,\hat{Z}_{\text{rel}}(u,v)}, \qquad \hat{Z}_{\text{metric}}(u,v) = s \cdot \hat{Z}_{\text{rel}}(u,v)$$

#### Mod A vs Mod B Eğim Hatası Yayılımı (20.000 Deneme, 5 Derece Gerçek Eğim, 1.5 m Pencere)

| 3B Geometrik Algılama Rejimi | İrtifa (AGL) | Eğim MAE (derece) | Eğim P95 Hatası (derece) | 5 Derece Padde Yanlış Ret (> 8 derece) | Operasyonel Güvenlik Kararı |
| :--- | :---: | :---: | :---: | :---: | :--- |
| Mod A: Birincil RGB-D (%0.5 derinlik, %2 poz gürültüsü) | 25 m | **0.17** | **0.42** | **%0.0** | Birincil onay kaynağı (yüksek doğruluk). |
| Mod B: `DAv2-S` + Kanonik + Rangefinder (%3 lokal, %5 ölçek) | 25 m | **1.53** | **3.75** | **%5.8** | Alçak irtifa aktif yörüngede kullanılabilir yedek. |
| Mod B: `DAv2-S` + Yalnızca Barometre (%6 lokal, %8 ölçek) | 80 m | **6.77** | **19.51** | **%54.3** | Yüksek irtifada kör iniş onayı **veto edilir**. |
| Mod B: `UAVScenes` Eğik Rejim (%10 lokal, %12 ölçek) | 130 m | **18.68** | **43.92** | **%81.2** | Yalnızca kaba aday tarama; alçalma zorunlu. |

Mod B etkinken 2.5B ızgaraya irtifa bağımlı ek geometrik varyans tabanı $\sigma^2_{\text{geo,mono}}(z_{\text{AGL}}) = 0.04 + (z_{\text{AGL}} / 60.0)^2 \cdot 0.25$ eklenir ve 30 m üzeri irtifada Mod B ile `COMMIT_TOUCHDOWN` yasaklanır [6, 7].

### C.3. 2.5B Bayesçi SLZ İnanç Izgarası Güncelleme Denklemleri

Yerel harita, araç merkezli kayan dairesel tampon olarak `80 m x 80 m` alanı `0.5 m` çözünürlükle (`160x160 = 25.600` hücre; 64 bayta hizalı yapı ile yalnızca **1.562 MB statik RAM**) tutar:

1. **Güvenilirlik Ağırlıklı Log-Odds Semantik Füzyon:** $p_{\text{safe}}(u) = P(y=0 \mid u) + P(y=1 \mid u) \in [10^{-3}, 1-10^{-3}]$, $l_{\text{obs}}(u) = \log\frac{p_{\text{safe}}(u)}{1 - p_{\text{safe}}(u)}$ ve $\tau_{\text{static}} = 30.0$ s için:
$$w_{\text{rel}}(u) = \frac{\cos\alpha_{\text{inc}}}{1 + \exp\!\left(\beta\,(\tilde{E}(u) - \tau_{\text{ood}})\right)}, \qquad w_{\text{eff}}(u) = \begin{cases} 2.0\,w_{\text{rel}}(u), & p_{\text{safe}}(u) < 0.5 \\ 1.0\,w_{\text{rel}}(u), & p_{\text{safe}}(u) \ge 0.5 \end{cases}$$
$$L_k(i,j) = \text{clip}\!\left(e^{-\Delta t / \tau_{\text{static}}}\,L_{k-1}(i,j) + w_{\text{eff},k}(i,j)\,l_{\text{obs},k}(i,j),\; -8.0,\; +8.0\right), \qquad P_{\text{safe},k}(i,j) = \frac{1}{1 + e^{-L_k(i,j)}}$$

2. **Dinamik Tehlikeler İçin Asimetrik Üstel Zamansal Sönümleme ($\tau_{\text{dyn}} = 5.0$ s):**
$$H_{\text{dyn},k}(i,j) = \max\!\left(p_{\text{dyn},k}(i,j),\; H_{\text{dyn},k-1}(i,j)\cdot \exp\!\left(-\frac{\Delta t}{\tau_{\text{dyn}}}\right)\right)$$

3. **Bakış Açısı Yeniliği İndirimli Ters-Varyans OOD Belirsizlik Yayılımı ($\rho_{\text{view}} \in [0.05, 1.0]$):**
$$\sigma^2_{\text{obs}}(u) = \sigma^2_{\text{base}}(z_{\text{AGL}}, \text{mode}) + \alpha_{\text{ood}}\,\left[\ln\!\left(1 + \exp\!\left(\frac{\tilde{E}(u) - \tau_{\text{ood}}}{\kappa}\right)\right)\right]^2$$
$$\Lambda_k(i,j) = \Lambda_{k-1}(i,j)\,e^{-\Delta t / \tau_{\text{var}}} + \frac{\rho_{\text{view}}(\Delta\mathbf{p}, \Delta\psi)}{\sigma^2_{\text{obs},k}(i,j)}, \qquad \sigma^2_{\text{cell},k}(i,j) = \frac{1}{\max(\Lambda_k(i,j),\; \Lambda_{\min})}$$

4. **Dairesel 5.0 m Rotor Açıklık Pad Toplulaştırması (317 Hücre, Kuyruk Yüzdelikleri):**
$$P_{\text{safe}}(\text{pad}) = \min\!\left(\text{Q}_{0.05}\!\left\{P_{\text{safe}}(c)\right\}_{c\in\mathcal{D}_R},\; 1 - \max_{c\in\mathcal{D}_R} H_{\text{dyn}}(c)\right)$$
$$\theta_{\max}(\text{pad}) = \max\!\left(\theta_{\text{disk}},\; \text{Q}_{0.95}\!\left\{\theta(c)\right\}_{c\in\mathcal{D}_R}\right), \qquad \sigma^2_{\text{pad}} = \text{Q}_{0.90}\!\left\{\sigma^2_{\text{cell}}(c)\right\}_{c\in\mathcal{D}_R}$$

---

## Bölüm D — Deterministik ~150 Satırlık C++20 Aktif Bilgi Toplama İnişi ve `ISlzPriorProvider` Arayüzü

### D.1. Aktif İnceleme Yörüngesi Durum Makinesi, Varyans Çöküşü ve Fiziksel Enerji Geçidi

Çevrimiçi POMDP ağaç araması ve RL uçuş kontrolü tamamen dışlanarak 4 durumlu, deterministik ve `noexcept` bir durum makinesi kurgulanmıştır:

| Hedef Durum | Deterministik Geçiş Koşulu | Eylem & Çıktı |
| :--- | :--- | :--- |
| `COMMIT_TOUCHDOWN` | `P_safe(pad) >= 0.85` VE `theta_max <= 8.0 derece` VE `sigma_z <= 0.15 m` VE `step_max <= 0.25 m` VE `R_clear >= 5.0 m` VE `H_dyn_max < 0.15` VE `sigma^2_pad < 0.08` VE (Mod B ise `AGL <= 30 m`) | TLOF merkezine son inişi onayla (`REASON_OK_COMMITTED`). Dinamik yaya/araç girişinde 20 Hz döngüde anında iptal/ret. |
| `ACTIVE_INSPECT_ORBIT` | Ön eleme geçerli (`P_safe >= 0.55`, `H_dyn_max < 0.30`, `theta_max <= 10.0 derece`, `sigma_z <= 0.22 m`, `R_clear >= 5.0 m`) VE yüksek varyans (`sigma^2_pad >= 0.08` veya Mod B irtifa vetosu) VE enerji geçidi (`SoC_current > SoC_reserve + Delta_SoC_orbit`) | `z_inspect = 22.0 m` AGL (`20–25 m`), `r_inspect = 15.0 m` (`12–18 m`) yarıçapla TLOF merkezine kilitli 150 derece (`120–180 derece`) yay uçuşu (`<= 12.0 s` bütçe). |
| `REJECT_DIVERT_NEXT` | Kesin tehlike (`P_safe < 0.40` veya `H_dyn_max >= 0.30` veya alçak irtifada `theta_max > 10.0 derece`), VEYA düşük varyansla ret, VEYA yetersiz enerji, VEYA 12.0 s zaman aşımı | Mevcut adayı kara listeye al ve `libvertirta` bir sonraki aday SLZ'ye yönlensin. |

**22 m AGL Aktif Yörünge Varyans Çöküşü ve Enerji Maliyeti (`VertiRTA-X8-240`, 240 kg, 13.2 kWh Batarya):** `z_inspect = 22.0 m` AGL ve `r_inspect = 15.0 m` yarıçap, yataydan 55.7 derece aşağı (34.3 derece nadir-dışı) bir bakış konisiyle parlamaları kırar ve hücre başına derinlik piksel yoğunluğunu 3.6'dan 48.1 piksele (~13.4x) çıkarır. Başlangıç varyansı `0.4500` olan bir çatı adayında 22 m yörünge boyunca pad varyansı `t = 2 s` -> `0.1260`, **`t = 4 s` -> `0.0733` (`< 0.08` ilk onay geçişi)**, `t = 6 s` -> `0.0516`, `t = 8 s` -> `0.0399`, `t = 10 s` -> `0.0325`, `t = 12 s` -> `0.0274` değerine çöker.

| Uçuş / Arıza Rejimi (240 kg Eş Eksenli X8, `A_disk = 8.042 m^2`, `k_coax = 1.22`, `FM = 0.68`, `eta = 0.86`) | Ortalama Güç (kW) | 12 s Enerji (Wh) | 12 s Maliyet (% SoC) |
| :--- | :---: | :---: | :---: |
| Nominal (8 rotor, sakin 0 m/s) | 53.7 | 178.9 | **%1.36** |
| Nominal (8 rotor, 10 m/s rüzgâr) | 60.1 | 200.3 | **%1.52** |
| Arızalı 1-Motor-Out (7 rotor, sakin) | 63.3 | 211.1 | **%1.60** |
| Arızalı 1-Motor-Out (7 rotor, 10 m/s rüzgâr) | 70.8 | 236.1 | **%1.79** |
| Arızalı 2-Motor-Out (6 rotor, 10 m/s rüzgâr) | 81.6 | 271.9 | **%2.06** |

*(Enerji güvenlik geçidi bu değerleri 1.25x emniyet çarpanıyla değerlendirir.)*

### D.2. C++20 `ISlzPriorProvider`, `IForwardHazardProvider` (V2.1 Taslağı) ve `ActiveInspectionPolicy`

```cpp
// include/vertirta/perception/slz_prior_provider.hpp
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <span>

namespace vertirta::perception {

enum class DepthMode : uint8_t { RGBD_PRIMARY = 0, MONO_FALLBACK = 1 };
enum class FaultMode : uint8_t { NOMINAL_8R = 0, DEGRADED_7R = 1, DEGRADED_6R = 2 };

enum class SlzDecisionState : uint8_t {
  CANDIDATE_EVAL = 0,
  ACTIVE_INSPECT_ORBIT = 1,
  COMMIT_TOUCHDOWN = 2,
  REJECT_DIVERT_NEXT = 3
};

enum SlzReasonBits : uint32_t {
  REASON_NONE           = 0u,
  REASON_OK_COMMITTED   = 1u << 0,
  REASON_HIGH_OOD_VAR   = 1u << 1,
  REASON_SLOPE_EXCEEDED = 1u << 2,
  REASON_ROUGH_EXCEEDED = 1u << 3,
  REASON_DYNAMIC_HAZARD = 1u << 4,
  REASON_LOW_SEM_SAFETY = 1u << 5,
  REASON_ENERGY_INSUFF  = 1u << 6,
  REASON_ORBIT_TIMEOUT  = 1u << 7,
  REASON_MONO_ALT_VETO  = 1u << 8
};

struct Vec3d { double x{0.0}, y{0.0}, z{0.0}; };

struct SlzCandidate {
  uint32_t id{0};
  Vec3d    center_enu{};        // TLOF merkezi (m, yerel ENU)
  float    p_safe{0.0f};        // Muhafazakâr disk güvenlik olasılığı [0, 1]
  float    h_dyn_max{1.0f};     // Disk içi maks dinamik tehlike yoğunluğu [0, 1]
  float    theta_max_deg{90.f}; // Maks yerel + disk düzlem eğimi (derece)
  float    sigma_z_m{1.0f};     // Düzlem pürüzlülüğü RMS (m)
  float    step_max_m{1.0f};    // Parapet / basamak maks dikey sapması (m)
  float    sigma2_pad{1.0f};    // Q90 epistemik / OOD pad varyansı
  float    clearance_r_m{0.0f}; // Doğrulanmış açık dairesel yarıçap (m)
  float    mean_ood_energy{0.f};
};

struct ActiveInspectSetpoint {
  Vec3d target_pos_enu{};
  Vec3d look_at_tlof_enu{};
  float desired_speed_mps{3.3f};
  float arc_progress_rad{0.0f};
  float elapsed_s{0.0f};
};

struct EnergyGateInput {
  float     soc_current_pct{50.0f};
  float     soc_reserve_pct{20.0f};
  float     wind_mps{0.0f};
  FaultMode fault_mode{FaultMode::NOMINAL_8R};
};

struct SlzProviderSnapshot {
  uint64_t                             timestamp_ns{0};
  DepthMode                            depth_mode{DepthMode::RGBD_PRIMARY};
  SlzDecisionState                     state{SlzDecisionState::CANDIDATE_EVAL};
  uint32_t                             reason_mask{REASON_NONE};
  std::optional<SlzCandidate>          active_candidate{};
  std::optional<ActiveInspectSetpoint> inspect_setpoint{};
  float                                inference_ms{0.0f};
};

class ISlzPriorProvider {
 public:
  virtual ~ISlzPriorProvider() = default;
  virtual SlzProviderSnapshot QuerySnapshot(const Vec3d& vehicle_enu,
                                            float agl_m,
                                            const EnergyGateInput& energy) noexcept = 0;
  virtual void ResetActiveCandidate() noexcept = 0;
};

// V2.1 Ayrıştırılmış İleri-Bakış İnce Tel / Engel Taslağı (512x512 nadir SLZ'den bağımsız)
struct ForwardWireHazard {
  Vec3d start_enu{};
  Vec3d end_enu{};
  float confidence{0.0f};
  float min_clearance_m{0.0f};
};

class IForwardHazardProvider {
 public:
  virtual ~IForwardHazardProvider() = default;
  virtual std::span<const ForwardWireHazard> QueryForwardHazards(uint64_t now_ns) const noexcept = 0;
};

// ================= Deterministik ~150-LOC Aktif İnceleme Yörüngesi Politikası =================
class ActiveInspectionPolicy {
 public:
  struct Config {
    float r_slz_min_m      = 5.0f;
    float p_safe_commit    = 0.85f;
    float p_safe_screen    = 0.55f;
    float p_safe_abort     = 0.40f;
    float theta_commit_deg = 8.0f;
    float theta_screen_deg = 10.0f;
    float sigma_z_commit_m = 0.15f;
    float sigma_z_screen_m = 0.22f;
    float step_max_m       = 0.25f;
    float h_dyn_commit_max = 0.15f;
    float h_dyn_abort_max  = 0.30f;
    float tau_commit       = 0.08f;
    float z_inspect_agl_m  = 22.0f;    // [20.0, 25.0] m
    float r_inspect_m      = 15.0f;    // [12.0, 18.0] m
    float arc_total_rad    = 2.61799f; // 150 derece ([120, 180] derece)
    float max_budget_s     = 12.0f;
    float soc_safety_mult  = 1.25f;
  };

  explicit ActiveInspectionPolicy(Config cfg = {}) noexcept : cfg_(cfg) {}

  [[nodiscard]] float EstimateOrbitDeltaSocPct(const EnergyGateInput& e) const noexcept {
    constexpr float kBaseSoc12s = 1.36f; // 53.7 kW nominal * 12 s / 13.2 kWh
    float fault_mult = 1.00f;
    if (e.fault_mode == FaultMode::DEGRADED_7R) fault_mult = 1.18f;
    if (e.fault_mode == FaultMode::DEGRADED_6R) fault_mult = 1.36f;
    const float wind_mult = 1.0f + 0.012f * std::clamp(e.wind_mps, 0.0f, 18.0f);
    return kBaseSoc12s * fault_mult * wind_mult * cfg_.soc_safety_mult;
  }

  SlzProviderSnapshot Step(const SlzCandidate& c,
                           const Vec3d& vehicle_enu,
                           float agl_m,
                           DepthMode mode,
                           const EnergyGateInput& energy,
                           uint64_t now_ns) noexcept {
    SlzProviderSnapshot out{};
    out.timestamp_ns = now_ns;
    out.depth_mode = mode;
    out.active_candidate = c;

    const bool geom_ok = (c.theta_max_deg <= cfg_.theta_commit_deg) &&
                         (c.sigma_z_m <= cfg_.sigma_z_commit_m) &&
                         (c.step_max_m <= cfg_.step_max_m) &&
                         (c.clearance_r_m >= cfg_.r_slz_min_m);
    const bool sem_ok  = (c.p_safe >= cfg_.p_safe_commit) && (c.h_dyn_max < cfg_.h_dyn_commit_max);
    const bool var_ok  = (c.sigma2_pad < cfg_.tau_commit);
    const bool mono_hi = (mode == DepthMode::MONO_FALLBACK) && (agl_m > 30.0f);

    if (geom_ok && sem_ok && var_ok && !mono_hi) {
      state_ = SlzDecisionState::COMMIT_TOUCHDOWN;
      out.state = state_;
      out.reason_mask = REASON_OK_COMMITTED;
      return out;
    }

    if (state_ == SlzDecisionState::ACTIVE_INSPECT_ORBIT && active_id_ == c.id) {
      const float elapsed_s = static_cast<float>(now_ns - orbit_start_ns_) * 1e-9f;
      if (c.p_safe < cfg_.p_safe_abort |

| c.h_dyn_max >= cfg_.h_dyn_abort_max)
        return Reject(out, REASON_DYNAMIC_HAZARD | REASON_LOW_SEM_SAFETY);
      if (c.theta_max_deg > cfg_.theta_screen_deg && agl_m <= 28.0f)
        return Reject(out, REASON_SLOPE_EXCEEDED);
      if (elapsed_s >= cfg_.max_budget_s)
        return Reject(out, REASON_ORBIT_TIMEOUT | REASON_HIGH_OOD_VAR);
      const float rem_soc = EstimateOrbitDeltaSocPct(energy) * (1.0f - elapsed_s / cfg_.max_budget_s);
      if (energy.soc_current_pct <= (energy.soc_reserve_pct + rem_soc))
        return Reject(out, REASON_ENERGY_INSUFF);
      out.state = SlzDecisionState::ACTIVE_INSPECT_ORBIT;
      out.reason_mask = mono_hi ? (REASON_HIGH_OOD_VAR | REASON_MONO_ALT_VETO) : REASON_HIGH_OOD_VAR;
      out.inspect_setpoint = ComputeOrbitSetpoint(c.center_enu, elapsed_s);
      return out;
    }

    const bool screen_ok = (c.p_safe >= cfg_.p_safe_screen) &&
                           (c.h_dyn_max < cfg_.h_dyn_abort_max) &&
                           (c.theta_max_deg <= cfg_.theta_screen_deg) &&
                           (c.sigma_z_m <= cfg_.sigma_z_screen_m) &&
                           (c.clearance_r_m >= cfg_.r_slz_min_m);
    const bool high_var_trigger = (!var_ok) |

| mono_hi;
    if (!screen_ok |

| !high_var_trigger) {
      uint32_t mask = REASON_NONE;
      if (c.p_safe < cfg_.p_safe_commit)           mask |= REASON_LOW_SEM_SAFETY;
      if (c.h_dyn_max >= cfg_.h_dyn_commit_max)    mask |= REASON_DYNAMIC_HAZARD;
      if (c.theta_max_deg > cfg_.theta_commit_deg) mask |= REASON_SLOPE_EXCEEDED;
      if (c.sigma_z_m > cfg_.sigma_z_commit_m)     mask |= REASON_ROUGH_EXCEEDED;
      return Reject(out, mask);
    }

    if (energy.soc_current_pct <= (energy.soc_reserve_pct + EstimateOrbitDeltaSocPct(energy)))
      return Reject(out, REASON_ENERGY_INSUFF | REASON_HIGH_OOD_VAR);

    state_ = SlzDecisionState::ACTIVE_INSPECT_ORBIT;
    active_id_ = c.id;
    orbit_start_ns_ = now_ns;
    start_azimuth_rad_ = static_cast<float>(
        std::atan2(vehicle_enu.y - c.center_enu.y, vehicle_enu.x - c.center_enu.x));
    out.state = state_;
    out.reason_mask = mono_hi ? (REASON_HIGH_OOD_VAR | REASON_MONO_ALT_VETO) : REASON_HIGH_OOD_VAR;
    out.inspect_setpoint = ComputeOrbitSetpoint(c.center_enu, 0.0f);
    return out;
  }

  void Reset() noexcept { state_ = SlzDecisionState::CANDIDATE_EVAL; active_id_ = 0; }

 private:
  SlzProviderSnapshot Reject(SlzProviderSnapshot& out, uint32_t mask) noexcept {
    state_ = SlzDecisionState::REJECT_DIVERT_NEXT;
    out.state = state_;
    out.reason_mask = mask;
    return out;
  }

  ActiveInspectSetpoint ComputeOrbitSetpoint(const Vec3d& tlof, float elapsed_s) const noexcept {
    const float frac = std::clamp(elapsed_s / cfg_.max_budget_s, 0.0f, 1.0f);
    const float az = start_azimuth_rad_ + frac * cfg_.arc_total_rad;
    return {
        {tlof.x + cfg_.r_inspect_m * std::cos(az),
         tlof.y + cfg_.r_inspect_m * std::sin(az),
         tlof.z + cfg_.z_inspect_agl_m},
        tlof,
        (cfg_.r_inspect_m * cfg_.arc_total_rad) / cfg_.max_budget_s,
        frac * cfg_.arc_total_rad,
        elapsed_s};
  }

  Config           cfg_{};
  SlzDecisionState state_{SlzDecisionState::CANDIDATE_EVAL};
  uint32_t         active_id_{0};
  uint64_t         orbit_start_ns_{0};
  float            start_azimuth_rad_{0.0f};
};

}  // namespace vertirta::perception
```

---

## Bölüm E — Sadece-Sensörlü Colab G4 Gazebo Harmonic EGL Hattı ve Çift Alanlı Değerlendirme Protokolü

### E.1. Sadece-Sensörlü Colab G4 Gazebo Harmonic EGL Kurulum Betiği (`< 2 dk` Drive Önbellek)

Bulut GPU konteynerlerinde `/usr/share/glvnd/egl_vendor.d/10_nvidia.json` mevcut olsa dahi `libEGL_nvidia.so.0` alt bağımlılıkları eksik kaldığında OGRE2 sessizce `llvmpipe` yazılım render motoruna düşebilir (`gz-sim#1272` [15]). Aşağıdaki betik sıfır PX4 ve sıfır ROS 2 bağımlılığıyla `gz-harmonic` paketlerini Drive önbelleğinden `< 2 dakika` içinde geri yükler ve NVIDIA EGL donanım hızlandırmasını doğrudan doğrular:

```bash
#!/usr/bin/env bash
# colab_g4_gz_egl_setup.sh — Sadece-Sensörlü Gazebo Harmonic (EGL) + < 2 dk Drive Önbellek
set -euo pipefail
CACHE_DIR="/content/drive/MyDrive/vertirta_cache"
CODENAME="$(lsb_release -cs)" # Colab: jammy (22.04) veya noble (24.04)
DEB_TAR="${CACHE_DIR}/gz_harmonic_${CODENAME}_debs.tar.zst"

# 1. Hızlı < 2 dk Drive önbellek geri yükleme (SIFIR PX4 / SIFIR ROS 2)
if [[ -f "${DEB_TAR}" ]]; then
  mkdir -p /tmp/gz_debs && tar --zstd -xf "${DEB_TAR}" -C /tmp/gz_debs
  sudo dpkg -i /tmp/gz_debs/*.deb |

| sudo apt-get install -f -y --no-download
else
  sudo apt-get update && sudo apt-get install -y curl lsb-release gnupg zstd mesa-utils-extra
  sudo curl -fsSL [https://packages.osrfoundation.org/gazebo.gpg](https://packages.osrfoundation.org/gazebo.gpg) \
       -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] \
    [http://packages.osrfoundation.org/gazebo/ubuntu-stable](http://packages.osrfoundation.org/gazebo/ubuntu-stable) ${CODENAME} main" \

| sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
  sudo apt-get update && sudo apt-get clean
  sudo apt-get install -y --download-only gz-harmonic libgz-transport13-dev libgz-msgs10-dev libgz-math7-dev
  mkdir -p "${CACHE_DIR}"
  (cd /var/cache/apt/archives && tar --zstd -cf "${DEB_TAR}" *.deb)
  sudo apt-get install -y gz-harmonic libgz-transport13-dev libgz-msgs10-dev libgz-math7-dev
fi

# 2. NVIDIA EGL Zorunlu Ortam Değişkenleri ve Donanım Doğrulama
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export DISPLAY=""
export GZ_PARTITION="vertirta_colab_g4"

EGL_NV_LIB="$(ldconfig -p | awk '/libEGL_nvidia\.so\.0/{print $NF; exit}')"
ldd "${EGL_NV_LIB}" | grep -q "not found" && { echo "HATA: Eksik NVIDIA EGL bağımlılığı!"; exit 1; }
eglinfo -B 2>/dev/null | grep -iE "NVIDIA|EGL_EXT_device_base" |

| { echo "HATA: NVIDIA EGL cihazı yok!"; exit 1; }

# 3. Headless OGRE2 EGL Sunucu Başlatma ve Yazılım Render Koruması
DISPLAY="" gz sim -v 4 -s -r --headless-rendering --render-engine-server ogre2 \
  /content/vertirta/sim/worlds/istanbul_maltepe_kartal.sdf > /tmp/gz_server.log 2>&1 &
sleep 4
grep -R "EGL Subsystem" ~/.gz/rendering/ogre2.log
grep -qi "llvmpipe" ~/.gz/rendering/ogre2.log && { echo "HATA: llvmpipe yazılım render tespit edildi!"; exit 1; }
gz topic -i --topic /slz_rgbd/image
```

### E.2. Gazebo Harmonic SDF Sensör Blokları ve C++20 `gz-transport` Köprüsü

```xml
<!-- istanbul_maltepe_kartal.sdf (Gazebo Harmonic / gz-sim8) -->
<world name="istanbul_coast">
  <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
    <render_engine>ogre2</render_engine>
  </plugin>
  <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
  <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>

  <!-- Örnek Etiketli Çatı (gz-sim-label-system ile segmentation kamerasına görünür) -->
  <model name="dragos_rooftop_01">
    <static>true</static>
    <link name="roof_slab">
      <visual name="slab_vis">
        <geometry><box><size>18.0 16.0 0.4</size></box></geometry>
        <plugin filename="gz-sim-label-system" name="gz::sim::systems::Label">
          <label>20</label> <!-- Kanonik 1: Safe_Flat_Roof -->
        </plugin>
      </visual>
    </link>
  </model>

  <!-- libvertirta 6-DOF tesisinin doğrudan konumlandırdığı kinematik sensör gövdesi -->
  <model name="vertirta_sensor_rig">
    <static>true</static>
    <link name="camera_link">
      <sensor name="slz_rgbd" type="rgbd_camera">
        <pose>0 0 -0.25 0 1.570796 0</pose> <!-- Nadir aşağı bakış (+90 derece pitch) -->
        <topic>slz_rgbd</topic>
        <always_on>1</always_on>
        <update_rate>20</update_rate>
        <visualize>false</visualize>
        <camera>
          <horizontal_fov>1.396263</horizontal_fov> <!-- 80 derece -> fx=fy=305.09 px @ 512x512 -->
          <image><width>512</width><height>512</height><format>R8G8B8</format></image>
          <clip><near>0.5</near><far>180.0</far></clip>
          <depth_camera><clip><near>0.5</near><far>150.0</far></clip></depth_camera>
        </camera>
      </sensor>
      <!-- Yalnızca 8.000 karelik veri üretiminde etkin -->
      <sensor name="slz_semantic" type="segmentation">
        <pose>0 0 -0.25 0 1.570796 0</pose>
        <topic>slz_semantic</topic>
        <always_on>1</always_on>
        <update_rate>20</update_rate>
        <visualize>false</visualize>
        <camera>
          <segmentation_type>semantic</segmentation_type>
          <horizontal_fov>1.396263</horizontal_fov>
          <image><width>512</width><height>512</height></image>
          <clip><near>0.5</near><far>180.0</far></clip>
        </camera>
      </sensor>
    </link>
  </model>
</world>
```

```cpp
// gz_egl_bridge.hpp (C++20, SPDX-License-Identifier: Apache-2.0)
#pragma once
#include <gz/msgs/boolean.pb.h>
#include <gz/msgs/image.pb.h>
#include <gz/msgs/pose.pb.h>
#include <gz/transport/Node.hh>
#include <atomic>
#include <cstring>
#include <mutex>
#include <vector>

class GzEglSensorBridge {
 public:
  bool Init(const std::string& world = "istanbul_coast") {
    set_pose_srv_ = "/world/" + world + "/set_pose/blocking";
    rgb_buf_.resize(512 * 512 * 3);
    depth_buf_.resize(512 * 512);
    bool ok1 = node_.Subscribe("/slz_rgbd/image", &GzEglSensorBridge::OnRgb, this);
    bool ok2 = node_.Subscribe("/slz_rgbd/depth_image", &GzEglSensorBridge::OnDepth, this);
    return ok1 && ok2;
  }

  // 6-DOF tesis adımından gelen kamera pozunu Gazebo EGL sahnesine senkron uygula
  bool StepSensorRigPose(double x, double y, double z,
                         double qw, double qx, double qy, double qz) {
    gz::msgs::Pose req;
    req.set_name("vertirta_sensor_rig");
    req.mutable_position()->set_x(x);
    req.mutable_position()->set_y(y);
    req.mutable_position()->set_z(z);
    req.mutable_orientation()->set_w(qw);
    req.mutable_orientation()->set_x(qx);
    req.mutable_orientation()->set_y(qy);
    req.mutable_orientation()->set_z(qz);
    gz::msgs::Boolean rep;
    bool result = false;
    return node_.Request(set_pose_srv_, req, 200u, rep, result) && result && rep.data();
  }

  bool CopyLatestRgbd(std::vector<uint8_t>& rgb_out, std::vector<float>& depth_out) {
    if (!has_rgb_.load() |

| !has_depth_.load()) return false;
    std::scoped_lock lk(mtx_);
    rgb_out = rgb_buf_;
    depth_out = depth_buf_;
    return true;
  }

 private:
  void OnRgb(const gz::msgs::Image& msg) {
    if (msg.data().size() != rgb_buf_.size()) return;
    std::scoped_lock lk(mtx_);
    std::memcpy(rgb_buf_.data(), msg.data().data(), rgb_buf_.size());
    has_rgb_.store(true);
  }
  void OnDepth(const gz::msgs::Image& msg) {
    if (msg.data().size() != depth_buf_.size() * sizeof(float)) return;
    std::scoped_lock lk(mtx_);
    std::memcpy(depth_buf_.data(), msg.data().data(), msg.data().size());
    has_depth_.store(true);
  }

  gz::transport::Node node_;
  std::string set_pose_srv_;
  std::mutex mtx_;
  std::vector<uint8_t> rgb_buf_;
  std::vector<float> depth_buf_;
  std::atomic<bool> has_rgb_{false}, has_depth_{false};
};
```

### E.3. 8.000 Karelik Maltepe–Dragos–Kartal Alan Rastgeleleştirmeli Veri Üreticisi

Operasyonel koridor, İstanbul Anadolu Yakası kıyı şeridinde **Maltepe Sahil KB (`40.9230 N, 29.1240 E`) -> Dragos Sahil (`40.9168 N, 29.1353 E`) -> Kartal Sahil GD (`40.8880 N, 29.1880 E`)** hattında **6.64 km** uzanır; **`LTFJ` (Sabiha Gökçen, `40.8986 N, 29.3092 E`) meydanının 10.26–15.80 km batısında** ve **`LTBX` (Samandıra, `40.9905 N, 29.2162 E`) meydanının 10.65–11.64 km güneybatısındadır**.

| Rastgeleleştirme Ekseni | Dağılım / Değer Aralığı | Kıyı İstanbul Fiziksel ve Algılama Gerekçesi |
| :--- | :--- | :--- |
| Topoğrafya & Yapı Katmanı | Copernicus DEM GLO-30 (`N40_E029`) + Marmara Denizi su düzlemi + 12 prosedürel kıyı bloğu *(OSM `ODbL` ayak izleri eğitim ağırlıklarından izole edilir)* | Maltepe dolgu parkı, Dragos tepesi eğimli yamaçları ve Kartal teras/kiremit çatılarının sıfır-ShareAlike temsili. |
| İrtifa & Kamera Yunuslama | 8.000 kare katmanlı: %35 alçak (15–30 m), %40 orta (30–65 m), %25 yüksek (65–120 m); pitch 0–35 derece nadir-dışı | İnceleme yörüngesi (22 m, 34.3 derece nadir-dışı) ile seyir taraması (60–120 m, nadir) arasındaki ölçek farkını kapsar. |
| Güneş, Gölge & Kıyı Sisi | Azimut 0–360 derece, güneş yüksekliği 12–72 derece, kıyı pus/sisi görüş 80–5000 m, Marmara speküler parıltısı | İstanbul kıyı şeridindeki sert bina gölgeleri ve deniz/çatı yansımaları altında OOD kalibrasyonu. |
| Çatı Tehlikeleri & OOD | Hareketli yaya/araç (`label=40`), HVAC/su deposu/parapet (`label=30`), %15 karede OOD branda/iskele/parıltı (`label=254`) | Dinamik tehlike tespiti ve enerji marjini aykırı değer eğitimi (OE). |
| Sensör Gürültüsü & Çıktılar | Kuadratik stereo derinlik gürültüsü (`sigma_Z = 0.002 * Z^2`), %2 derinlik kaybı, 1–5 px hareket bulanıklığı -> `PNG RGB`, `float32 depth`, `float16 normal`, `uint8 5-sınıf+OOD maske`, `JSON pose` | Gerçek RGB-D bozulmalarına dayanıklı sim-to-real aktarımı. |

### E.4. Dürüst Çift Alanlı Kıyaslama ve 3 Yollu Kapalı Döngü Ablasyon Protokolü

#### Tablo 1 — Sıfır-Atışlı Gerçek Dünya Değerlendirme Şeması (Eval-Only Karantina Kıyaslamaları)

| Eval-Only Kıyaslama (Asla Eğitime Girmez) | Değerlendirme Metrikleri | Kapı 5 Hedef Eşiği | `PIDNet-S` (Damıtmasız Taban) | `PIDNet-S` (+DINOv2+DAv2+OE) | Protokol Notu |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `AeroScapes` (`CC BY-SA 4.0`, 3.269 kare, 5–30 m) | 4-Sınıf mIoU / SLZ F1 / False-Safe Rate (FSR, düşük iyi) | mIoU >= %62, SLZ F1 >= %74, FSR <= %4.0 | `[Ölçülecek]` | `[Ölçülecek]` | `Safe_Flat_Roof` AeroScapes'te yok; mevcut 4 kanonik sınıf üzerinden makro değerlendirme. |
| `ICG Semantic Drone` (`NC`, 400 açık kare, 5–50 m) | 5-Sınıf mIoU / Çatı-Zemin SLZ F1 / FSR (düşük iyi) | mIoU >= %60, SLZ F1 >= %75, FSR <= %3.5 | `[Ölçülecek]` | `[Ölçülecek]` | `6000x4000` nadir görüntüler `512x512` oran korumalı ölçekleme ile değerlendirilir. |
| OOD Ayırma (`ICG` `conflicting/obstacle` + `Gazebo OOD`) | OOD AUROC (yüksek iyi) / FPR@95 (düşük iyi) (Serbest Enerji vs MaxLogit) | AUROC >= 0.88, FPR@95 <= %25 | `[Ölçülecek]` | `[Ölçülecek]` | Sınır-telafili Serbest Enerji ile ham `MaxLogit` yan yana kıyaslanır. |
| `UAVScenes` (`CC BY-NC-SA 4.0`, RGB + Livox Avia LiDAR, 80–130 m) | Mod B `DAv2-Small` Baro-Çapalı Eğim MAE (derece) / AbsRel vs LiDAR GT | 80 m: AbsRel <= 0.12, Eğim MAE <= 7.0 derece; alçak kesitlerde <= 2.5 derece | `[Ölçülecek: Ham DAv2-S]` | `[Ölçülecek: Kanonik+Baro]` | LiDAR yer gerçeği düzlem eğimine karşı Mod B'nin yüksek irtifa sınırlarını belgeler. |

#### Tablo 2 — Canlı Colab-G4 6-DOF + Gazebo EGL 3 Yollu Kapalı Döngü Ablasyonu (`N >= 60`) ve Yerel WSL2 Duman Doğrulaması

Her senaryoda (`N = 60` rastgeleleştirilmiş tohum: 20 temiz düz çatı, 20 dinamik çatı yayası/aracı enjeksiyonu, 20 OOD branda/enkaz + sert gölge/sis) üç politika aynı başlangıç pozu, rüzgâr ve arıza profiliyle kıyaslanır:

| Politika / Doğrulama Hattı | Güvenli İniş % (yüksek iyi) | Yanlış Onay / False-Commit % (düşük iyi) | Aktif Yörünge Tetikleme % | Ek Enerji Maliyeti (% SoC) | Donanım Etiketli C++20 ORT Gecikmesi (p99) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| (1) Kör DEM-Only SLZ (Colab G4 6-DOF + EGL, `N = 60`) | `[Ölçülecek]` | `[Ölçülecek]` | %0.0 | %0.00 | `N/A` (Görsel ağ yok) |
| (2) Pasif Tek-Geçişli Görsel SLZ (Colab G4 6-DOF + EGL, `N = 60`) | `[Ölçülecek]` | `[Ölçülecek]` | %0.0 | %0.00 | Colab G4 `RTX PRO 6000 Blackwell` (`TRT FP16`): `[Ölçülecek]` *(Hedef `< 2.0 ms`)* |
| (3) Aktif Belirsizlik-Yörüngeli SLZ (Colab G4 6-DOF + EGL, `N = 60`) | `[Ölçülecek]` *(Kapı 5: >= %90)* | `[Ölçülecek]` *(Kapı 5: <= %2.0)* | `[Ölçülecek]` *(Hedef %30–%55)* | `[Ölçülecek]` *(Fizik üst sınırı: <= %2.06)* | Colab G4 `RTX PRO 6000 Blackwell` (`TRT FP16`): `[Ölçülecek]` *(Hedef `< 2.0 ms` ağ / `< 8 ms` toplam)* |
| Yerel WSL2 Duman Doğrulaması (Ubuntu 24.04 + PX4 v1.16 + ROS 2 Jazzy + `ros-jazzy-ros-gz`) | `3/3` geçiş | `0/3` | Doğrulandı | `< %2.0` | Yerel `GTX 1650 (TU117, 4 GB)` (`ORT-CUDA FP32` vs `FP16`): `[Ölçülecek]` *(Hedef `< 12 ms` ağ, `< 300 MB` VRAM, `20–30+ Hz`)* |

> **Yerel WSL2 Duman Doğrulama Notu:** WSL2 üzerinde EGL başsız sürücü kararsızlıklarına karşı `px4io/px4-sitl-gazebo-ros2` (Ubuntu 24.04 + ROS 2 Jazzy + Gazebo Harmonic) veya yerel Jazzy ortamı standart WSLg OpenGL (`QT_QPA_PLATFORM=xcb`) ile çalıştırılır; `GTX 1650 (TU117)` üzerinde Tensor Çekirdeği bulunmaması nedeniyle `ORT-CUDA FP32` ve `FP16` gecikmeleri karşılaştırmalı olarak raporlanır.

---

## Doğrulanmış 2024–2026 ve Temel Teknik Kaynaklar

1. **Yang et al. (2024).** *Depth Anything V2.* NeurIPS 2024 (`arXiv:2406.09414`; `github.com/DepthAnything/Depth-Anything-V2`; ViT-S Apache-2.0, ViT-B/L/G CC-BY-NC-4.0).
2. **Hoffman Lab (2024).** *SkyScenes: A Synthetic Dataset for Aerial Scene Understanding.* ECCV 2024 (`huggingface.co/datasets/hoffman-lab/SkyScenes`, MIT Lisansı).
3. **Yin et al. (2023) & Hu et al. (2024).** *Metric3D / Metric3D v2: A Versatile Monocular Geometric Foundation Model for Zero-Shot Metric Depth and Surface Normal Estimation.* (`arXiv:2404.15506`).
4. **Juergens, M., Meinert, N., Bengs, V., Hüllermeier, E., & Waegeman, W. (2024).** *Is Epistemic Uncertainty Faithfully Represented by Evidential Deep Learning?* (`arXiv:2402.09056`) & **Shen et al. / Wornell (2024).** *Are Uncertainty Quantification Capabilities of Evidential Deep Learning a Mirage?*
5. **Wang, S., Li, S., Zhang, Y., Yu, S., Yuan, S., She, R., Xie, L., & Tay, W. P. (2025).** *UAVScenes: A Multi-Modal Dataset for UAVs.* ICCV 2025 (`github.com/sijieaaa/UAVScenes`, CC BY-NC-SA 4.0 — Eval-Only).
6. **Song, Z., Chen, G., Zhang, Y., Zou, Y., Fu, C., Yuan, Z., Huang, C., Cui, S., & Cao, X. (2026).** *AerialMetric: Benchmarking and Adapting UAV Monocular Metric Depth Estimation in the Real World.* ECCV 2026, pp. 360–380 (`arXiv:2606.29716`).
7. **ISPRS Archives (2026).** *Transferring Depth Anything V2 from Terrestrial to Aerial UAV Imagery (MDE4BH).* Vol. XLIX-B4-2026, p. 679 (13 Temmuz 2026).
8. **IEEE Open Journal of the Computer Society (2026).** *A Vision-Based Framework for Safe Landing Zone Mapping of UAVs in Dynamic Environments.* Vol. 7, pp. 492–503 (Ocak 2026).
9. **OR-SLZNet (2026).** *Ordinal Regression Safe Landing Zone Detection Combining RGB, Depth, Flatness, and Surface Inclination.* IEEE Computer Society (2026).
10. **Barbato et al. (2026).** *FlyAwareV2: A Multimodal Cross-Domain UAV Dataset for Urban Scene Understanding.* Signal Processing: Image Communication, 117483 (`GPL-3.0` + `UAVid` nedeniyle elendi).
11. **Meta AI Research (2023–2025).** *DINOv2 (`arXiv:2304.07193`, Apache-2.0)* vs *DINOv3 License (`LICENSE.md`, 19 Ağustos 2025 — ITAR/Askerî yasaklar nedeniyle kesin yasaklı)*.
12. **Xu, Xiong, & Bhattacharyya (2023).** *PIDNet: A Real-Time Semantic Segmentation Network Inspired by PID Controllers.* CVPR 2023 (`github.com/XuJiacong/PIDNet`, MIT) & **Hong et al. (2022).** *DDRNet.* IEEE T-ITS 24(3):3448–3460 (`github.com/ydhongHIT/DDRNet`, MIT).
13. **Liu, W., Wang, X., Owens, J., & Li, Y. (2020).** *Energy-Based Out-of-Distribution Detection.* NeurIPS 2020; **Tian et al. (2022).** *PEBAL.* ECCV 2022; **Nayal, Yavuz, Henriques, & Güney (2023).** *RbA.* ICCV 2023.
14. **İzinli & Denetlenen Hava Veri Setleri:** IGN France *FLAIR-1 / FLAIR-2* (`etalab-2.0`); Xia et al. (2023) *OpenEarthMap* (`zenodo.org/records/7223446`, `open-earth-map.org/attribution.html`); Rahnemoonfar et al. (2021) *FloodNet-Supervised v1.0* (`CDLA-Permissive-1.0`); Rahnemoonfar et al. (2023) *RescueNet* (*Scientific Data* 10:913; resmi veri lisansı `CC BY-NC-ND 4.0`).
15. **Çalışma Zamanı & Simülasyon Belgeleri (2024–2026):** Open Robotics Gazebo Harmonic (`gz-sim8`, EGL/OGRE2, `gz-sim#1272`, `gz-sensors#545`), Microsoft ONNX Runtime (`CUDA 12.8/13.0`, `OrtTensorRTProviderOptionsV2`, `IoBinding`), ESA Copernicus DEM GLO-30 Açık Lisansı ve 2025 IFP Kılavuzu.