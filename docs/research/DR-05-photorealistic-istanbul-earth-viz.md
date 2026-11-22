# VertiRTA-X8-240 İstanbul Anadolu Yakası Fotogerçekçi Dünya Görselleştirme ve Uçuş Tekrarı Mimarisi (Maltepe–Dragos–Kartal–LTFJ/LTBX)

`VertiRTA-X8-240` (240 kg koaksiyel eVTOL) platformunun İstanbul Maltepe–Dragos–Kartal kıyı koridorunda, gerçek `LTFJ` (Sabiha Gökçen) ve `LTBX` (Samandıra) CTR hava sahası duvarlarına bitişik arıza ve acil iniş senaryolarını %100 kullanım koşullarına (ToS) uyumlu ve fotogerçekçi biçimde görselleştirmek için **İki Katmanlı (Dual-Tier) Hibrit Mimari** kilitlenmiştir: (1) yerel Windows iş istasyonunda (NVIDIA GTX 1650, 4 GB VRAM) canlı 3B operasyon ve telemetri denetimi için **CesiumJS (`:5180`)**, (2) bulut GPU ortamında (`Google Colab G4 — NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB GDDR7 ECC`) `libvertirta` ikili uçuş kaydedicisinin (`.vrta`) doğrudan sürdüğü **Başsız (Headless) Blender 5.2 LTS Cycles OptiX** çevrimdışı 4K sinema hattı. Bu mimari karar; İstanbul'da Google Photorealistic 3D Tiles 3B fotogrametri ağının bulunmaması, Unreal Engine 5 ile NVIDIA Isaac Sim'in 4 GB yerel VRAM ve Colab kapsayıcı taşınabilirliğini kırması, 3D Gaussian Splatting (3DGS) için gereken hava çekimlerinin SHGM SHT-İHA mevzuatı uyarınca `LTFJ`/`LTBX` 5 NM yasak bölgelerinde hukuken yapılamaması ve ham Three.js kutularının `float32` hassasiyet titremesi ile gerçek arazi eksikliği nedeniyle elenmesiyle belirlenmiştir.

Mimari, ticari ve hukuki uyumluluğu korumak adına canlı akış ile çevrimdışı işleme (offline rendering) arasında kesin bir veri güvenlik duvarı uygular. Bing Maps Aerial, Esri World Imagery, Cesium World Terrain ve Cesium OSM Buildings **yalnızca `:5180` tarayıcı istemcisinde resmi canlı akış API'leriyle** tüketilir; hiçbir ticari karo (tile) diske kaydedilmez, kazınmaz veya Blender dokusuna gömülmez. Çevrimdışı 4K Blender Cycles hattı ise bütünüyle serbest, açık ve ticari kullanıma uyumlu **Copernicus DEM GLO-30** (`EGM2008` düşey datumu, `N40_00_E029_00` COG), **Copernicus Sentinel-2 L2A** (`35TPF` MGRS karosu, 10 m RGB+NIR), **OpenStreetMap + Overture Maps** conflate edilmiş bina tabanları/yolları, **İBB CC BY 4.0** mahalle kat istatistikleri ve **DHMİ eAIP Türkiye** resmi hava sahası koordinatları üzerine kurulur. Sentinel-2'nin 10 m yer örnekleme aralığı 1 m AGL yakın çatı inişlerinde tek başına yetersiz kaldığından, yakın mesafede gerçekçilik bütünüyle prosedürel Türk kıyı mimarisi PBR gölgelendiricileri (kiremit ve betonarme çatılar, parapet/klima detayları, kıyı asfaltı/parkları ve Marmara speküler deniz suyu) ile sağlanır.

Üretime geçişte üç kritik teknik kısıt ve çekince tasarım aşamasında kilit altına alınmıştır. İlk olarak, **Blender 4.2 LTS ve 4.3 sürümleri NVIDIA Blackwell (`sm_120`) Cycles OptiX çekirdeklerini desteklemez**; Blackwell desteği Blender 4.4 ile eklenmiş olup Kasım 2026 itibarıyla üretim sürümü olarak **Blender 5.2 LTS** (NVIDIA sürücü >= 575), sürücü geride kalırsa yedek olarak **Blender 4.5 LTS** (sürücü >= 535) sabitlenmelidir. İkinci olarak, popüler açık kaynak türevlerinden **EOX Sentinel-2 Cloudless 2018–2025** mozaikleri ve **FABDEM** çıplak yer modeli **CC BY-NC-SA 4.0 (ticari olmayan)** lisanslıdır; ticari saflık için doğrudan açık Copernicus lisanslı AWS `sentinel-cogs` L2A bantları ve bina tabanı düzleştirilmiş GLO-30 kullanılmalıdır. Üçüncü olarak, Kartal Kıyı Vertiportu (`40,8860° K, 29,1880° D, 12,0 m AMSL`) merkezli 15 x 15 km sahada yalnızca İstanbul'un yaklaşık `+36,7 m` `EGM2008` geoit ondülasyonunu değil, düzlem ENU'dan küresel ECEF'e geçerken Maltepe'de `+3,56 m` ve alan köşesinde `+8,82 m` havada asılı kalma hatası yaratan Dünya eğriliği düşüşünü de kapatan eğrilik-telafili dönüşüm sözleşmesi uygulanmalıdır.

---

## 1. Doğrulanmış ve ToS-Temiz İstanbul Coğrafi Veri Yığını ve Hukuki Sınırlar

İstanbul Anadolu Yakası'nda gerçek dünya tabanlı bir eVTOL dijital ikizi kurmanın ilk aşaması, küresel 3B harita sağlayıcılarının Türkiye kapsamındaki fiili boşluklarını ve yerel belediye verilerinin erişim sınırlarını netleştirmektir. Google Maps Platform, Photorealistic 3D Tiles ürününün 49'dan fazla ülkede 2.500'ü aşkın şehri kapsadığını ve Google Earth ile aynı eğik hava fotogrametrisi kaynağını kullandığını belirtmektedir. Buna karşın Google'ın resmi kapsam belgeleri (`developers.google.com/maps/coverage`) Photorealistic 3D Tiles için şehir bazlı bir kullanılabilirlik listesi yayımlamamaktadır. Bağımsız geliştirici doğrulamaları, Google Haritalar/Earth Yardım Topluluğu ürün uzmanı açıklamaları ve topluluk kayıtları, ulusal hava görüntüleme izin politikaları nedeniyle **İstanbul ve Türkiye genelinde Google Photorealistic 3D Tiles 3B fotogrametri ağının bulunmadığını**; API'nin İstanbul üzerinde yalnızca kaba arazi yüzeyine giydirilmiş düz 2B uydu görüntüsü döndürdüğünü ortaya koymaktadır.

Kapsam boşluğu giderilse dahi Google Map Tiles API politikaları (9 Kasım 2026 güncellemesi) bu veri kaynağını çevrimdışı uçuş tekrarı boru hattından hukuken dışlar. Söz konusu politikalar harita karolarının çevrimdışı kullanımını, toplu indirilmesini, HTTP `Cache-Control` ve `ETag` üstbilgileri ötesinde önbelleklenmesini, uydu görüntüsünden 3B bina veya arazi modeli türetilmesini ve makine/görüntü analizi amacıyla işlenmesini açıkça yasaklamaktadır. Ek olarak Map Tiles API içeriği barındıran tanıtım videoları **en fazla 30 saniye** ile sınırlandırılmış olup açıkça "yalnızca tanıtım amaçlıdır" ibaresini taşımak zorundadır; bu hüküm 60 saniyelik 4K mühendislik uçuş testi filmlerini doğrudan geçersiz kılar. Avrupa Ekonomik Alanı (EEA) faturalandırma adresine bağlı projelerde ise 8 Temmuz 2025 sonrasında oluşturulan veya değiştirilen uygulamalar için Photorealistic 3D Tiles ve 2B Uydu karoları **HTTP 403** hatasıyla tamamen kapatılmıştır (Türkiye faturalı hesaplar EEA dışındadır ancak Avrupalı paydaşlarla aynı API anahtarı paylaşılamaz).

Yerel düzeyde İstanbul Büyükşehir Belediyesi (İBB), hava LiDAR taramaları, hava fotoğrafları ve 1:1.000 halihazır haritaları birleştirerek — 2019 tarihli teknik yayına göre yaklaşık 1,5 milyon LoD2 bina küpü, 3.800 LoD3 simge yapı ve 135 TB depolama hacmine ulaşan — bir 3B kent modeli geliştirmiş; 2023–2024 faaliyet raporlarında bu CityGML verisini Esri SLPK biçimine dönüştürerek **`3bistanbul.ibb.gov.tr`** web uygulamasında etkileşimli biçimde yayımladığını bildirmiştir. Ancak İBB Açık Veri Portalı (`data.ibb.gov.tr`) CC BY 4.0 lisansı altında *Mahalle Bazlı Bina Analiz Verisi* (2016–2017 MAKS ve afet güncellemeleri) ile yeşil alan katmanları gibi tablosal/2B veri setlerini sunmasına karşın, `3bistanbul.ibb.gov.tr` arkasındaki ham 3B CityGML veya SLPK bina ağları için kamusal bir toplu indirme (bulk download) erişimi bulunmamaktadır. Bu nedenle belediye web uygulamasının akış karolarını kazımadan, bütünüyle açık lisanslı küresel kaynaklarla kendi deterministik 3B İstanbul kentsel modelimizi inşa etmek zorunludur.

### Katı Hukuki ve ToS Güvenlik Duvarı (Katman 1 Canlı Akış vs. Katman 2 Çevrimdışı İşleme)

Hukuki güvenliği sağlamak amacıyla tüm coğrafi kaynaklar iki geçirimsiz katmana ayrılmıştır. **Katman 1 (Yalnızca `:5180` Tarayıcı Canlı Akışı — CesiumJS)** resmi streaming API'leriyle Cesium World Terrain, Cesium OSM Buildings ve yüksek çözünürlüklü uydu katmanlarını yalnızca anlık görüntüleme için tüketir. Cesium ion Arşiv ve Dışa Aktarma kuralları uyarınca, kullanıcının kendisinin yüklemediği Asset Depot varlıkları (**Cesium World Terrain** ve **Cesium OSM Buildings**) kesinlikle indirilemez veya çevrimdışı dışa aktarılamaz. Benzer biçimde Microsoft Bing Maps Platform API Kullanım Koşulları (1 Nisan 2024 güncellemesi) **Madde 2(c)** ile statik harita görsellerini birleştirerek yeni görüntü ürünü oluşturmayı veya özellik sayısallaştırmayı, **Madde 5(n)** ile içeriğin kopyalanmasını, saklanmasını, arşivlenmesini veya veritabanı kurulmasını yasaklar. Esri Master Agreement (E300) de standart **Esri World Imagery** katmanının çevrimdışı dışa aktarım, türev üretim ve önbellekleme için kullanılamayacağını ve ticari uygulamalarda anahtarsız (keyless) doğrudan bağlantıya izin verilmediğini hükme bağlar. (Esri'nin ayrı *World Imagery for Export* servisi 150.000 karoya kadar çevrimdışı dışa aktarım sunsa da kurumsal/geliştirici ArcGIS aboneliği gerektirir ve yalnızca ArcGIS / ArcGIS Runtime SDK uygulamaları içinde kullanım için lisanslanmıştır; Blender Cycles çevrimdışı dokusu için kullanılamaz.)

Katman 1 işletiminde Kasım 2026 itibarıyla iki ek Bing Maps kısıtı gözetilmelidir. Birincisi, **Cesium ion Ek B-1 (20 Ağustos 2025 güncellemesi)** uyarınca Cesium ion üzerinden akan Bing Maps içeriği **işletme varlık takibi (asset tracking), filo yönetimi veya sevk/idare** amacıyla kullanılamaz; tekil Ar-Ge uçuş tekrarı bu kapsama girmese de canlı çoklu araç operasyonları için API anahtarlı Esri ArcGIS Location Platform veya Azure Maps altlığına geçilmelidir. İkincisi, Microsoft ücretsiz Bing Maps for Enterprise anahtarlarını **30 Haziran 2025** tarihinde sonlandırmış olup kurumsal sözleşmelerin nihai kapanış tarihi **30 Haziran 2028**'dir; Cesium ion, kendi kurumsal anlaşması sayesinde Bing katmanını geçiş döneminde sürdürmekle birlikte 2026 sonrası için Google Maps 2D Tiles ve Azure Maps gibi alternatifleri devreye almaktadır. Bu nedenle `:5180` istemcisinde baz uydu katmanı, tek bir yapılandırma parametresiyle Esri World Imagery, Cesium ion Bing veya Azure Maps arasında değiştirilebilir soyutlamayla tasarlanır.

**Katman 2 (%100 ToS-Temiz Çevrimdışı İşleme — Colab G4 Blender Cycles)** ise yerel diske ve bulut GPU belleğine yasal olarak indirilebilen, ticari kullanıma açık, sıfır kazıma gerektiren kaynaklarla beslenir. Bu katmanda iki yaygın açık veri lisans tuzağından bilinçli biçimde kaçınılmıştır. Birincisi **EOX Sentinel-2 Cloudless 2018–2025** tuzağıdır: EOX IT Services'in 2018–2024 ve 2025 bulutsuz mozaikleri (`s2maps.eu`) **CC BY-NC-SA 4.0 (Ticari Olmayan)** lisanslıdır ve ticari kullanım ücretli *EOX Commercial Attribution-RestrictedUse 1.1* lisansı gerektirir; ticari olarak ücretsiz tek sürüm olan 2016 mozaiği (`CC BY 4.0`) ise 10 yıl eskidir ve Kartal/Maltepe kıyı dolgularıyla yeni yapılaşmayı yansıtmaz. Çözüm, AWS Open Data üzerindeki **`s3://sentinel-cogs/sentinel-s2-l2a-cogs/`** deposundan (Element 84 Earth Search STAC API) güncel 2025/2026 yaz dönemi bulutsuz **`35TPF`** MGRS karosunu (10 m B04/B03/B02 RGB + B08 NIR) doğrudan ticari kullanıma ve türev üretime izin veren **AB/ESA Copernicus Serbest, Tam ve Açık Veri Politikası** kapsamında indirmektir. İkincisi **FABDEM** çıplak yer modeli tuzağıdır: Copernicus GLO-30 bir Dijital Yüzey Modeli (DSM) olduğundan kentsel alanlarda ortalama 1,61 m bina yansıma yanlılığı içerir; ancak bu yanlılığı temizleyen FABDEM türevi **CC BY-NC-SA 4.0** lisanslı olduğundan Fathom'dan ticari lisans alınmadan ticari projelerde kullanılamaz. Ticari temizliği korumak için birincil yöntem olarak açık lisanslı Copernicus GLO-30 üzerine OSM/Overture bina tabanları maskelenerek morfolojik taban düzleştirmesi uygulanır (isteğe bağlı olarak, topluluk kaynaklarında `CC BY 4.0` lisanslı 30 m çıplak yer DTM'i olarak bildirilen OpenGeoHub **GEDTM30** yerel İstanbul kıyı/vertiport doğrulaması yapılmak kaydıyla ikincil referans olarak değerlendirilebilir).

> **Şekil 1: İki Katmanlı Mimari Canlı Akış ile Çevrimdışı 4K İşleme Arasında Hukuki Güvenlik Duvarı Kurar**
> Tekil **.vrta** ikili uçuş kaydı iki bağımsız görselleştirme hattını besler. Üst katman yerel Windows GTX 1650 (4 GB VRAM) üzerinde yalnızca resmi canlı akış API'lerini kullanırken, alt katman Colab G4 Blackwell (96 GB VRAM) üzerinde bütünüyle açık lisanslı Copernicus, OSM/Overture ve DHMİ verileriyle çevrimdışı 4K sinema üretir; kırmızı güvenlik duvarı ticari karoların çevrimdışı aktarılmasını engeller.

### İstanbul 15 x 15 km Koridor Geometrisi ve Doğrulanmış Varlık Matrisi (Teslimat 1)

Kartal Kıyı Vertiportu (`40,8860° K, 29,1880° D, 12,0 m AMSL`) yerel ENU orijini `(E = 0, N = 0, Z = 0)` olarak kilitlendiğinde, WGS84 elipsoidi üzerinde Maltepe–Dragos–Kartal kıyısını ve `LTFJ`/`LTBX` sınır duvarlarını kapsayan iki 15,00 x 15,00 km sınır kutusu (BBOX) hesaplanmıştır. **Kanonik Simetrik Kutu** ($E, N \in [-7.5, +7.5]$ km) doğrudan Kartal merkezli olup `W = 29,099011° D, S = 40,818464° K, E = 29,276989° D, N = 40,953536° K` (UTM Zone 35N `EPSG:32635`: `677.019–691.640 m D`, `4.520.725–4.536.096 m K`) sınırlarına oturur; Maltepe Orhangazi Şehir Parkı, Dragos Tepesi, Kartal Vertiportu, `LTFJ` batı CTR duvarı, `LTFJ`/`LTBX` güneybatı birleşim köşesi ve `LTBX` güney CTR duvarı eksiksiz biçimde kutunun içindedir. **Opsiyonel Koridor-Kaydırılmış Kutu** (`40,9050° K, 29,2050° D` merkezli; `W = 29,116011° D, S = 40,837464° K, E = 29,293989° D, N = 40,972536° K`, yerel ENU orijini yine Kartal) ise Marmara açıklığından pay kırparak doğuda LTFJ 06/24 pisti batı eşiğine (`29,2950° D`) kadar uzanır. Her iki kutu da **tek bir Copernicus GLO-30 karosuna (`N40_00_E029_00`)** ve **tek bir Sentinel-2 MGRS karosuna (`35TPF`)** tam olarak sığmaktadır.

Resmi DHMİ eAIP Türkiye belgelerinden çıkarılan koordinatların Kartal Vertiportu ENU çerçevesine aktarılması, bu koridorun neden yüksek yoğunluklu bir acil iniş test sahası olduğunu sayısal olarak gösterir. LTFJ ARP (`40°53'54"K, 029°18'33"D`), Kartal Vertiportu'nun `10,302 km (5,563 NM)` doğu-kuzeydoğusundadır; 5 NM (`9,260 km`) yarıçaplı ve `2.500 FT AMSL (762,0 m)` tavanlı LTFJ CTR batı duvarı Kartal Vertiportu'nun **yalnızca 1,04 km doğu-kuzeydoğusunda** yükselir. LTBX CTR (`40°59'56"K, 029°12'58"D` merkezli, 5 NM yarıçaplı) ise Kartal'ın `12,758 km (6,889 NM)` kuzey-kuzeydoğusundadır ve iki CTR'nin güneybatı ortak köşesi (`40°54'46,03"K, 029°11'31,64"D`) Kartal'ın **yalnızca 3,00 km kuzeyinde** (`dE = +347 m, dN = +2.975 m`) konumlanır. Not olarak, meydana özgü yetkili belge `LT_AD_2_LTBX_en.pdf` (AD 2.17) ve `LT_AD_2_LTFM_VFR_en.pdf` harita lejandı Samandıra CTR üst sınırını **`SFC – 2.000 FT AMSL (609,6 m)`** verirken, LTFM VFR haritasındaki bir özet satırında çevreleyen İstanbul TMA tabanıyla örtüşen `1.500 FT AMSL` ifadesi yer almaktadır; görselleştirme mimarisi yetkili meydan AIP'si doğrultusunda LTBX perdesini **`2.000 FT AMSL`** olarak kilitler ve `1.500 FT (457,2 m)` seviyesinde belirgin bir TMA taban çizgisi çizer.

| Hava Sahası / Referans Noktası | Resmi DHMİ eAIP Koordinatı (WGS84) | Kartal ENU Konumu (`dE, dN`) | Kartal'a Mesafe | Düşey Limit / Rakım (`AMSL` -> `WGS84 HAE`) |
| :--- | :--- | :--- | :--- | :--- |
| **Kartal Kıyı Vertiportu (Orijin)** | `40,886000° K, 29,188000° D` | `0 m, 0 m` | `0,00 km` | `12,0 m AMSL` -> `~48,68 m HAE` |
| **LTFJ (Sabiha Gökçen) ARP & CTR** | `40°53'54"K, 029°18'33"D` (5 NM) | `+10.210 m, +1.377 m` | `10,30 km` (`CTR duvarı: 1,04 km`) | `SFC – 2.500 FT (762,0 m AMSL)` -> **`798,58 m HAE`** |
| **LTBX (Samandıra) CTR Merkezi** | `40°59'56"K, 029°12'58"D` (5 NM; ARP `40°59'26"K`) | `+2.365 m, +12.537 m` | `12,76 km` | `SFC – 2.000 FT (609,6 m AMSL)` -> **`646,35 m HAE`** |
| **LTFJ / LTBX Güneybatı Köşesi** | `40°54'46,03"K, 029°11'31,64"D` | `+347 m, +2.975 m` | `3,00 km` | LTFJ/LTBX ortak kesim ucu (KD ucu: `40°59'15"K, 029°19'30"D`) |
| **Dragos Tepesi / Maltepe Sahil SLZ** | `40,9125° K, 29,1460° D` / `40,9225° K, 29,1240° D` | `-3.540, +2.940 m` / `-5.390, +4.050 m` | `4,60 km` / `6,75 km` | Aday acil iniş çatı ve açık park sahaları |

> **Şekil 2: Maltepe–Dragos–Kartal Koridoru LTFJ Batı Duvarının 1 km Batısında Sıkışık Hava Sahasında İşler**
> Kartal Kıyı Vertiportu (yerel ENU orijini) merkezli 15 x 15 km simülasyon sahası; Maltepe kıyısı, Dragos Tepesi ve Marmara Denizi'ni kapsarken doğuda **LTFJ CTR (5 NM, 2500 FT)** duvarına yaklaşık 1 km, kuzeyde ise **LTFJ/LTBX** ortak sınır köşesine 3 km mesafede konumlanır.

Aşağıdaki varlık matrisi, her iki görselleştirme katmanı için doğrulanmış erişim uç noktalarını, lisans rejimlerini, çözünürlükleri ve düşey datumları sabitlemektedir:

| Veri Katmanı / Varlık | Resmi Kaynak ve Erişim URL / S3 Yolu | Lisans ve Ticari Uyum | Çözünürlük / Hassasiyet | Yatay CRS / Düşey Datum | Mimari Rol ve ToS Sınırı |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Copernicus DEM GLO-30 (`COP-DEM-GLO-30-F`)** | AWS Open Data (`--no-sign-request`, `eu-central-1`): `s3://copernicus-dem-30m/Copernicus_DSM_COG_10_N40_00_E029_00_DEM/Copernicus_DSM_COG_10_N40_00_E029_00_DEM.tif` veya ESA CDSE / OpenTopography `OTSDEM.032021.4326.3` | Copernicus WorldDEM-30 Free, Full & Open (DLR/Airbus atfı + sorumluluk reddi zorunlu; ticari & çevrimdışı OK) | 1" (~30,85 m K–G x 23,41 m D–B @ 40,89° K); mutlak < 4 m LE90 (< 3 m %95 EEA39), bağıl < 2 m (eğim <= %20) | `EPSG:4326` (WGS84) / **`EGM2008` (`EPSG:3855`)**; `RasterPixelIsPoint` | **Katman 2 & Gazebo:** Birincil arazi yükseklik ağı; bina tabanları ve vertiport pedleri düzleştirilerek kullanılır. |
| **Sentinel-2 L2A (`35TPF` MGRS Karosu)** | AWS `s3://sentinel-cogs/sentinel-s2-l2a-cogs/` (Element 84 Earth Search STAC: `earth-search.aws.element84.com`) | EU/ESA Copernicus Legal Notice (tamamen açık, ticari, önbellek & çevrimdışı OK; atıf zorunlu) | 10 m GSD (B02/B03/B04 RGB + B08 NIR); 2025–2026 bulutsuz yaz sahnesi | `EPSG:32635` (UTM Zone 35N) / Yatay ortofoto | **Katman 2:** Makro kıyı renk tabanı + NDVI (park/ağaç) ve NDWI (Marmara kıyı maskesi) prosedürel karıştırma girdisi. |
| **OpenStreetMap Vektörleri (Yol, Kıyı, Bina, Park)** | Geofabrik Türkiye Özeti: `https://download.geofabrik.de/europe/turkey-latest.osm.pbf` (~623–654 MB, günlük güncel) | **ODbL 1.0** (4K video/kareler *Produced Work* sayılır, `© OpenStreetMap contributors` atfı yeterlidir; türev vektör DB paylaşılırsa ODbL gerekir) | Vektör (~0,5–2 m yatay); `building:levels`, `height`, `highway`, `natural=coastline`, `leisure=park` | `EPSG:4326` / Göreli bina yüksekliği (`m` AGL) | **Katman 2 & Gazebo:** Yol şeritleri, Marmara kıyı çizgisi, parklar ve birincil bina tabanları (`tile.openstreetmap.org` kazınmaz). |
| **Overture Maps Buildings (`2026-11-18.0`+)** | AWS S3 GeoParquet (DuckDB): `s3://overturemaps-us-west-2/release/<latest>/theme=buildings/type=building/*.parquet` (STAC: `stac.overturemaps.org/catalog.json`) | **ODbL 1.0** (OSM içeren bina teması) / **CDLA Permissive 2.0** (GERS ID ve OSM dışı kaynaklar) | Vektör çokgenler (OSM + Microsoft ML + Google Open Buildings conflation); `height`, `num_floors`, `roof_shape` | `EPSG:4326` / Göreli yapı yüksekliği (`m` AGL) | **Katman 2 & Gazebo:** OSM'deki eksik bina ayak izlerini ve kat/yükseklik boşluklarını doldurur. |
| **İBB Mahalle Bazlı Bina Analiz Verisi** | İBB Açık Veri Portalı: `https://data.ibb.gov.tr/dataset/mahalle-bazli-bina-analiz-verisi` | **CC BY 4.0** (ticari kullanım ve atıfla serbest) | Mahalle düzeyi kat dağılımı (Maltepe, Dragos, Kartal; 2016–2017 MAKS / 2023 İBB) | Tablosal istatistiksel öncül (prior) | **Katman 2:** Hem OSM hem Overture'da `height`/`num_floors` eksik binalar için deterministik yükseklik atama kalibrasyonu. |
| **DHMİ eAIP Türkiye Hava Sahası (LTFJ & LTBX)** | `dhmi.gov.tr/AIPDocuments/LT_AD_2_LTFJ_en.pdf`, `LT_AD_2_LTBX_en.pdf`, `LT_AD_2_LTFM_VFR_en.pdf` | Resmi kamusal havacılık yayını (eAIP Türkiye) | LTFJ ARP `40°53'54"K 029°18'33"D` (5 NM, `2.500 FT`); LTBX CTR `40°59'56"K 029°12'58"D` (5 NM, `2.000 FT`) | `WGS84` / **`AMSL`** (`LTFJ` geoit ondülasyonu `120 FT = 36,58 m`) | **Katman 1 & 2:** 3B kırmızı/kehribar CTR hava sahası perdeleri ve 500 FT irtifa çizgileri. |
| **Cesium World Terrain + OSM Buildings** | Cesium ion SaaS (`Cesium.Terrain.fromWorldTerrain()`, `createOsmBuildingsAsync()`) | Cesium ion ToS (Ek A & B; ODbL atfı zorunlu; **çevrimdışı dışa aktarım yasak**) | 30–90 m Quantized-Mesh 1.0 + global 3B OSM binaları | `WGS84 ECEF` / **`WGS84 HAE` (`EPSG:4979`)** | **Yalnızca Katman 1 (Canlı `:5180`):** Tarayıcı içi canlı 3B küre ve arazi zemini. |
| **Bing Maps Aerial / Esri World Imagery / Azure Maps** | Cesium ion Bing Asset (2026–2028 geçişli) veya API anahtarlı Esri `ArcGisMapServerImageryProvider.fromUrl` | Bing TOU §2(c), §5(n), ion Ek B-1 (filo takibi yasak) / Esri E300 (**kazıma & çevrimdışı yasak**) | ~0,3–1,0 m kentsel uydu/hava görüntüsü | `EPSG:3857` (Web Mercator akış karoları) | **Yalnızca Katman 1 (Canlı `:5180`):** Canlı tarayıcı görüntüsü; Blender'a aktarılması kesinlikle yasaktır. |

---

## 2. Dört Yönlü Görselleştirme Motoru Karşılaştırması (Teslimat 2)

Motor seçimi iki zıt donanım gerçeğini aynı anda çözmek zorundadır: mühendisin masasındaki **Windows / NVIDIA GTX 1650 (4 GB VRAM, RT çekirdeksiz)** üzerinde etkileşimli gerçek-dünya uçuş denetimi ve **Google Colab G4 (Linux, NVIDIA RTX PRO 6000 Blackwell 96 GB VRAM)** üzerinde tamamen betiklenebilir, sıfır GUI'li 4K sinema üretimi. Aşağıdaki karşılaştırma tablosu, önerilen İki Katmanlı Yığın ile üç ana endüstriyel alternatifi teknik ve hukuki sınırlar açısından kıyaslamaktadır.

| Değerlendirme Kriteri | **Önerilen: CesiumJS (`:5180`) + Colab Blender 5.2 LTS Cycles OptiX** | **Cesium for Unreal Engine 5 (UE5 + MRQ)** | **NVIDIA Isaac Sim 5.1/6.0 + Cesium for Omniverse** | **3D Gaussian Splatting (3DGS)** |
| :--- | :--- | :--- | :--- | :--- |
| **İstanbul 3B Gerçekçiliği ve Kapsam** | **Yüksek:** Canlıda 0,3–1 m Bing/Esri + OSM 3B binalar; 4K sinemada GLO-30 + Sentinel-2 L2A + Overture/OSM + prosedürel Türk kıyı PBR çatıları. | **Orta–Düşük (Ek Geliştirmesiz):** İstanbul'da Google 3D Mesh yine yoktur; 1 m AGL çatı inişi için prosedürel PCG/materyal hattını UE5 içinde sıfırdan kurmak gerekir. | **Orta–Düşük:** İstanbul'da yine Google 3D Mesh eksik; USD sahnesine OpenStreetMap/prosedürel kentsel boru hattı inşa etmek ağır mühendislik yükü getirir. | **Hukuken İmkansız (%0 Kapsam):** 15 x 15 km İstanbul koridorunda açık 3DGS taraması yoktur; `LTFJ`/`LTBX` 5 NM CTR içinde İHA çekimi **SHGM SHT-İHA** uyarınca yasaktır. |
| **Yerel Windows GTX 1650 (4 GB VRAM) Uyumu** | **Tam Uyum:** CesiumJS WebGL2 bellek sınırlaması (`cacheBytes: 256 MB`, `SSE: 24`, `requestRenderMode: true`) ile GPU belleğini < 1,8 GB hedefinde tutarak 30 Hz telemetride kararlı çalışır. | **Başarısız / Sınırda:** UE5 önerilen 8+ GB VRAM ve 32 GB RAM ister; Lumen/MegaLights için RTX 2000+ (DX12 SM6) gerekir; kullanıcı raporlarında MRQ `r.forceLOD 0` ile 11–15+ GB VRAM tüketip çökmektedir. | **Tamamen Çalışmaz:** Isaac Sim RT çekirdeksiz GPU'ları reddeder; asgari gereksinim **RTX 4080 / 16 GB VRAM** ve 32 GB RAM'dir. | **Başarısız:** Şehir ölçekli 3DGS sahneleri > 10–66 GB bellek gerektirir; özel hiyerarşik streaming olmadan 4 GB VRAM'e sığmaz. |
| **Colab G4 Linux Başsız Taşınabilirliği** | **Yüksek (< 60 sn kurulum):** Tek 368 MiB `.tar.xz` Blender 5.2 LTS arşivi; `blender -b --cycles-device OPTIX` X11/Vulkan takas zinciri gerektirmeden doğrudan çalışır. | **Çok Zayıf (Çok Haftalık Yük):** 50–100+ GB Linux motor/DDC kurulumu, Epic/GitHub kaynak hesabı, Windows-only eklenti uyumsuzluğu ve her oturumda uzun gölgelendirici derlemesi. | **Kırık (Colab Uyumsuz):** Docker + NVIDIA Container Toolkit (Colab içinde desteklenmez), 50+ GB disk, Ubuntu 24.04'te GCC 11 zorunluluğu; başsız GUI `vkCreateSwapchainKHR` hatasıyla çöker. | **Eğitim Darboğazı:** MatrixCity 2,7 km^2 sentetik kentte 23,7M Gaussian (~8,8M/km^2) raporlanmıştır; 456 B/ilkel maliyetiyle 120–225 km^2 alana ölçekleme ~1–2 milyar Gaussian (~450+ GB) olup tek 96 GB GPU'da ham eğitim sığmaz. |
| **Kararlılık ve 2025–2026 Ekosistem Olgunluğu** | **Üretim Seviyesi:** Blender 5.2 LTS (Temmuz 2028 destekli) RTX PRO 6000 Workstation'da #1 Cycles puanına (17.306–18.024) sahiptir; CesiumJS WebGL standardıdır. | **Orta:** Linux başsız MRQ (`-RenderOffscreen`) akışında OOM hataları ve Cesium eklentisinde MRQ viewport boyut raporlama aksaklıkları belgelenmiştir. | **Kırılgan:** `cesium-omniverse` Kit 107 -> 109 -> 110 geçişlerinde kırılmalar yaşamaktadır; **Issue #783** (v0.29.0 başsız başlatma deadlock'u) ve **#765** (yükseklik sorgu eksikliği) açıktır. | **Araştırma Aşaması:** Hava nadir görüntülerinden kentsel 3DGS geniş düz çatılarda geometrik bozulma, gömülü sabit ışık (yeniden aydınlatılamaz) ve yansıtıcı su artefaktı üretir. |
| **ToS ve Yasal Uyumluluk** | **%100 Temiz:** Katman 1 yalnızca resmi canlı akış; Katman 2 yalnızca açık Copernicus + OSM/Overture + DHMİ verisi. | **Riskli (Çevrimdışı Baskıda):** UE5 MRQ ile Bing/Google/ion karolarından 60 sn 4K video üretmek Google'ın 30 sn ve Bing/Esri'nin çevrimdışı/türev yasaklarıyla çatışır. | **Riskli:** Cesium ion üzerinden akan üçüncü taraf ticari harita karolarından çevrimdışı video üretimi aynı üçüncü taraf ToS sınırlarına tabidir. | **Yasadışı:** SHGM SHT-İHA Madde 18 gereği `LTFJ`/`LTBX` pist kenarından 5 NM (9 km) içinde ve kalabalık alanlarda izinsiz İHA uçuşu 2920 Sayılı Kanun kapsamında yaptırıma tabidir. |
| **Mühendislik Süresi ve `.vrta` Entegrasyonu** | **Düşük (Günler):** Python `bpy` doğrudan `.vrta` ikili dosyasını okuyup anahtar kare, rotor bulanıklığı, duman ve kamera kurgusu oluşturur; `:5180` WebSocket anında bağlanır. | **Yüksek (Haftalar):** C++ eklenti derleme, Sequencer/MRQ otomasyonu, Linux paketleme ve özel prosedürel materyal hattı gerektirir. | **Orta–Yüksek (Haftalar):** Omniverse Kit/USD şeması, ayrı cloud VM barındırma (Docker) ve başsız deadlock yamaları gerektirir. | **Uygulanamaz:** Yasal izin + 225 km^2 tarama + çoklu-GPU eğitim aylarca sürer ve `LTFJ` yaklaşma konisinde uçuş izni alınamaz. |

### Ham Three.js Kutularının Sayısal Reddi ve Blender Blackwell Sürüm Kilidi

Ham Three.js sahneleri yalnızca görsel gerçekçilikten yoksun kutular ürettiği için değil, temel bir GPU donanım kısıtı nedeniyle de elenmektedir. WebGL gölgelendiricileri **32-bit kayan nokta (`float32`, ~7 anlamlı basamak)** ile çalışır; WGS84 ECEF yer merkezli koordinat ölçeğinde ($R \approx 6\,378\,137$ m), `float32` temsil hatası metre düzeyinde köşe ve kamera titremesine (jitter) neden olur. Bu sorunu gidermek için Relative-To-Center (RTC) veya çift `float32` bölmeli Relative-To-Eye (RTE) özel gölgelendiriciler, logaritmik derinlik tamponu ve quantized-mesh/3D Tiles akış motoru sıfırdan yazılmak zorundadır — bu da CesiumJS'in yıllardır üretim kalitesinde sunduğu çekirdek altyapıyı çok daha zayıf biçimde yeniden icat etmek demektir.

Çevrimdışı 4K sinema hattında Blender sürüm seçimi donanım uyumu açısından hayati önem taşır. **Blender 4.2 LTS ve 4.3, NVIDIA Blackwell mimarisi (`RTX 50` serisi ve `RTX PRO 6000 Blackwell`) için Cycles OptiX çekirdek desteğine sahip değildir**; Blackwell desteği resmi olarak **Blender 4.4** ile eklenmiştir. Kasım 2026 itibarıyla iki paralel uzun vadeli destek (LTS) hattı mevcuttur: Birincil öneri olan **Blender 5.2 LTS** (14 Temmuz 2026 çıkışlı, Temmuz 2028'e kadar destekli, 5.2.4 Linux x64 arşivi 368 MiB, gömülü Python 3.13) Cycles doku önbelleği, yansız hacimler ve ACES/HDR renk uzayı sunar; ilk 5.0 sürümünde toplulukça bildirilen performans gerilemeleri 5.1'deki %5–10 GPU hız kazanımı ve 5.2 LTS ile aşılmış olup Open Data'da RTX PRO 6000 puanı 5.2.0'da (17.306) 4.5.0'ın (16.737) önüne geçmiştir. Ancak Blender 5.2 OptiX için **NVIDIA sürücü >= 575** şart koştuğundan, Colab G4 oturumunun `535–574` aralığında bir sürücüyle açılması halinde yedek sürüm olarak **Blender 4.5 LTS** (Temmuz 2027'ye kadar destekli, sürücü >= 535, gömülü Python 3.11) otomatik devreye alınmalıdır. (10 Kasım 2026'da yayımlanan Blender 5.3'ün DLSS 4.5 Ray Reconstruction özelliği sürücü >= 620 istediğinden bulut sürücü gecikmelerine karşı üretim hattı 5.2 LTS'de sabit tutulur.)

---

## 3. Kesin Jeodezik ve Düşey Datum Hizalama Sözleşmesi (Teslimat 3)

Headless Gazebo Harmonic'te Kartal veya Dragos çatısında `0,00 m AGL` olarak gerçekleşen bir acil inişin hem CesiumJS'te hem de Blender Cycles'ta **ne 36,7 m yer altına gömülmesi ne de 1,7–8,8 m havada asılı kalması** için dört alt sistemin çerçeve ve düşey datum sözleşmesi aşağıda matematiksel olarak kilitlenmiştir.

| Alt Sistem | Dünya Çerçevesi | Gövde Çerçevesi | Düşey Yükseklik Referansı | Kritik Hizalama Riski |
| :--- | :--- | :--- | :--- | :--- |
| **Headless Gazebo Harmonic** | Yerel Düzlem **ENU** (`+X` Doğu, `+Y` Kuzey, `+Z` Yukarı) | **FLU** (`+X` İleri, `+Y` Sol, `+Z` Yukarı) | Copernicus GLO-30 **`EGM2008` (`EPSG:3855`)**; `Z_gz = H_AMSL - 12,0 m` | `SphericalCoordinates::LOCAL` (eski hatalı) yerine `LOCAL2` kullanılmalı; düzlem DEM eğrilik içermez. |
| **PX4 Autopilot (`uORB`)** | Yerel **NED** (`+X` Kuzey, `+Y` Doğu, `+Z` Aşağı) | **FRD** (`+X` İleri, `+Y` Sağ, `+Z` Aşağı) | `alt` = AMSL (m), `alt_ellipsoid` = WGS84 HAE (m), `dist_bottom` = AGL (m) | MAVLink/PX4 içinde EGM96/EGM2008 karışıklığı oluşabilir; `.vrta` yalnızca `EGM2008` kilitler. |
| **CesiumJS (`:5180`)** | Küresel **WGS84 ECEF** (`X, Y, Z`) ve `(lat, lon, h)` | glTF 2.0 (Y-Yukarı otomatik Z-Yukarıya, `+X` İleri / FLU) | **`WGS84` Elipsoit Yüksekliği (`HAE`, `EPSG:4979`)** — deniz seviyesi != 0 | Ham AMSL girilirse araç ~36,7 m yer altında kalır; CWT ile GLO-30 arasında yerel 1–4 m fark çıkabilir. |
| **Blender 5.2 LTS (`bpy`)** | Yerel Sağ-El **ENU** (`+X` Doğu, `+Y` Kuzey, `+Z` Yukarı; `1 BU = 1 m`) | **FLU** (`+X` İleri, `+Y` Sol, `+Z` Yukarı) | **`EGM2008` (`EPSG:3855`)** — Gazebo ile **özdeş** GLO-30 ağı (`Z_bl = H_AMSL - 12,0 m`) | Gazebo ile birebir aynı DEM ve bina yüksekliği kullanıldığında temas sapması `0,000 m` olur. |

### İstanbul Geoit Ondülasyonu ve Dünya Eğriliği Tuzağı

Resmi DHMİ eAIP meydan verileri İstanbul bölgesindeki geoit ondülasyonunu üç bağımsız noktada doğrulamaktadır: **LTFM** (İstanbul Havalimanı, `41°16'31"K, 028°45'07"D`) için `37,19 m` (`122 FT`), **LTBA** (Atatürk, `40°58'34"K, 028°48'51"D`) için `36,88 m` (`121 FT`) ve **LTFJ** (Sabiha Gökçen, `40°53'54"K, 029°18'33"D`) için `36,58 m` (`120 FT`). Bu üç resmi istasyondan Kartal Kıyı Vertiportu'na (`40,8860° K, 29,1880° D`) yapılan doğrusal düzlem enterpolasyonu `36,62 m` (aşağıdaki sayısal testlerde nominal referans `+36,68 m` / `~+36,7 m`) değerini ve 15 x 15 km alan boyunca kuzeye doğru `+0,85 cm/km`, doğuya doğru `-0,54 cm/km` gradyan (köşegen boyunca toplam `21,4 cm` değişim) üretir. Çalışma zamanında ise sabit tahmin yerine doğrudan PROJ `us_nga_egm08_25.tif` / `egm2008-1` gridinden noktasal sorgu yapılır. (Harita Genel Müdürlüğü / Kılıçoğlu vd. Türkiye çalışmasına göre EGM2008 ile Antalya ortalama deniz seviyesine bağlı yerel Türk nivelman ağı arasında ortalama 0,86–0,89 m sabit ofset bulunsa da, Copernicus GLO-30, PROJ `EPSG:9518` ve Cesium ion dönüşümleri ortak biçimde **EGM2008** kullandığından tüm alt sistemler kesin olarak EGM2008'e sabitlenir.)

Yalnızca `+36,7 m` geoit farkını eklemek tek başına yeterli değildir; çünkü düzlem simülatörlerden küresel görünüm motoruna geçerken **ikinci gizli düşey hata (Dünya eğriliği düşüşü)** ortaya çıkar. Kartal Vertiportu enleminde (40,8860° K) WGS84 elipsoidinin asal dikey eğrilik yarıçapı $N_{\text{curv}} = 6\,387\,303.50$ m, meridyen eğrilik yarıçapı $M_{\text{curv}} = 6\,362\,794.00$ m ve Gauss ortalama yarıçapı $R_{\text{gauss}} = 6\,375\,036.97$ m'dir. Gazebo'nun düzlem ENU yükseklik haritasındaki $(E, N, Z_{\text{flat}})$ konumu eğrilik düzeltmesi yapılmadan Kartal'daki teğet ECEF düzlemine aktarılırsa, Dünya yüzeyi teğet düzlemin altında $(E^2 + N^2) / (2 R_{\text{gauss}})$ kadar aşağı kıvrıldığından araç CesiumJS'te **Dragos Tepesi'nde (4,60 km) `+1,66 m`, Maltepe Sahili'nde (6,75 km) `+3,56 m` ve 15 x 15 km kutu köşesinde (10,61 km) `+8,82 m` havada asılı kalır**.

> **Şekil 3: Eğrilik Telafisiz Düzlem ENU Maltepe'de 3,56 m, Alan Köşesinde 8,82 m Havada Kalma Hatası Üretir**
> Üst panel dört hata senaryosundaki mutlak düşey sapmayı karşılaştırır (ham AMSL'nin **36,68 m** yeraltı gömülmesi ve düzlem ENU'nun **8,82 m** köşe sapması vs. sözleşmemizin **< 0,009 m** hassasiyeti). Alt panel, Kartal Vertiportu'ndan uzaklaştıkça düzlem ENU ile eğrilik-telafili dönüşümün düşey hatasını gösterir.

### Kesin Matematiksel Dönüşüm Zinciri

#### 1. Eğrilik-Telafili ENU <-> WGS84 ECEF ve EGM2008 Düşey Dönüşümü

WGS84 parametreleri ($a = 6\,378\,137.0$ m, $f = 1 / 298.257223563$, $e^2 = 2f - f^2$) ile herhangi bir jeodezik noktanın $(\phi, \lambda, h_{\text{HAE}})$ ECEF vektörü ve Kartal Vertiportu orijinindeki (`40,8860° K, 29,1880° D`, $H_0 = 12.0$ m AMSL, $h_0 = H_0 + N_0 \approx 48.68$ m, $\mathbf{p}_0 = [4215767.7064, 2354954.3336, 4152892.1696]^T$ m) yerel ENU -> ECEF dönme matrisi şu biçimdedir:

$$N_{\nu}(\phi) = \frac{a}{\sqrt{1 - e^2 \sin^2\phi}}, \qquad \mathbf{p}_{\text{ECEF}} = \begin{bmatrix} (N_{\nu}(\phi) + h_{\text{HAE}})\cos\phi\cos\lambda \\ (N_{\nu}(\phi) + h_{\text{HAE}})\cos\phi\sin\lambda \\ (N_{\nu}(\phi)(1 - e^2) + h_{\text{HAE}})\sin\phi \end{bmatrix}$$

$$\mathbf{R}_{\text{ECEF}}^{\text{ENU}}(\phi_0, \lambda_0) = \begin{bmatrix} -\sin\lambda_0 & -\sin\phi_0\cos\lambda_0 & \cos\phi_0\cos\lambda_0 \\ \cos\lambda_0 & -\sin\phi_0\sin\lambda_0 & \cos\phi_0\sin\lambda_0 \\ 0 & \cos\phi_0 & \sin\phi_0 \end{bmatrix}$$

Gazebo/Blender düzlem ENU konumundan $(E, N, Z_{\text{flat}})$ CesiumJS küresel koordinatlarına geçerken önce eğrilik-telafili teğet düşey bileşen hesaplanır ve 8 adımlı yinelemeli ECEF -> Jeodezik ters dönüşümüyle ($< 10^{-9}$ m gidiş-dönüş hatası) anlık enlem-boylam $(\phi, \lambda)$ bulunur:

$$U_{\text{tan}} = Z_{\text{flat}} - \frac{E^2 + N^2}{2 R_{\text{gauss}}}, \qquad \mathbf{p}_{\text{ECEF}}^{\text{corr}} = \mathbf{p}_0 + \mathbf{R}_{\text{ECEF}}^{\text{ENU}}(\phi_0, \lambda_0) \begin{bmatrix} E \\ N \\ U_{\text{tan}} \end{bmatrix}$$

Ardından yerel geoit gradyanını da sıfırlayan kesin WGS84 elipsoit yüksekliği ve nihai ECEF konumu atanır:

$$H_{\text{AMSL}} = H_0 + Z_{\text{flat}}, \qquad h_{\text{WGS84}} = H_{\text{AMSL}} + N_{\text{EGM2008}}(\phi, \lambda)$$

Bu sözleşme sayesinde tüm 15 x 15 km alanda jeodezik konumlama hatası **< 9 mm** sınırında kalır. CTR tavanları da aynı formülle CesiumJS'e aktarılır: `LTFJ 2.500 FT AMSL` (`762,00 m`) -> **`798,58 m WGS84 HAE`** ($N = 36.58$ m) ve `LTBX 2.000 FT AMSL` (`609,60 m`) -> **`646,35 m WGS84 HAE`** ($N \approx 36.75$ m).

#### 2. PX4 (`NED/FRD`) -> Gazebo/Blender (`ENU/FLU`) -> CesiumJS (`ECEF`) Kuaterniyon Zinciri

Resmi `px4_ros_com/frame_transforms.h` tanımlarına uygun olarak Hamilton $[w, x, y, z]$ biçimindeki sabit çerçeve dönüşümleri $q_{\text{ENU}\leftarrow\text{NED}} = [0, \sqrt{2}/2, \sqrt{2}/2, 0]$ ve $q_{\text{FRD}\leftarrow\text{FLU}} = [0, 1, 0, 0]$ şeklindedir. PX4'ün gövdeden dünyaya yönelim kuaterniyonu $q_{\text{NED}\leftarrow\text{FRD}} = (w, x, y, z)$ için Gazebo/Blender `ENU/FLU` yönelimi ve CesiumJS küresel yönelimi kapalı biçimde şu şekilde hesaplanır (dört referans yönelim testinde $10^{-12}$ hassasiyetle doğrulanmıştır):

$$q_{\text{ENU}\leftarrow\text{FLU}} = q_{\text{ENU}\leftarrow\text{NED}} \otimes q_{\text{NED}\leftarrow\text{FRD}} \otimes q_{\text{FRD}\leftarrow\text{FLU}} = \frac{1}{\sqrt{2}}\begin{bmatrix} w + z \\ x + y \\ x - y \\ w - z \end{bmatrix}, \qquad \mathbf{R}_{\text{ECEF}\leftarrow\text{FLU}} = \mathbf{R}_{\text{ECEF}}^{\text{ENU}}(\phi, \lambda)\,\mathbf{R}\!\left(q_{\text{ENU}\leftarrow\text{FLU}}\right)$$

#### 3. Sıfır Santimetre (`0,00 m AGL`) Çatı ve Vertiport Temas Sözleşmesi

Gazebo Harmonic ile Blender Cycles arasında tam temas (`0,000 m` sapma), her iki motorun kırpılmış `Copernicus_DSM_COG_10_N40_00_E029_00_DEM` ağını ve aynı deterministik tohumla üretilmiş Overture+OSM bina yükseklik tablosunu birebir paylaşmasıyla garanti edilir. Canlı CesiumJS (`:5180`) tarafında ise Cesium World Terrain + Cesium OSM Buildings yüzeyleri Copernicus GLO-30 + Overture yüksekliklerinden yerel olarak `1–4 m` farklılık gösterebilir. Bunu tamamen gidermek için iki aşamalı mekanizma kurulur: Tercih edilen kurulumda kırpılmış GLO-30 GeoTIFF'i Cesium ion'a `EGM2008 / Mean Sea Level` düşey datumuyla özel arazi karosu olarak yüklenir ve aday SLZ çatı/vertiport glTF pedleri kesin koordinatlarına sabitlenir. Varsayılan Cesium World Terrain + OSM Buildings akışında ise hedef SLZ kilitlendiğinde CesiumJS'in hem araziyi hem 3D Tiles bina çatılarını örnekleyen `scene.sampleHeightMostDetailed([Cartographic])` çağrısıyla görsel yüzeyin gerçek elipsoit yüksekliği $h_{\text{cesium,surf}}$ ölçülür, artık yüzey farkı $\delta z_{\text{surf}} = h_{\text{cesium,surf}} - (H_{\text{sim,surf}} + N_{\text{EGM2008}})$ hesaplanır ve son yaklaşmada yumuşakça sönümlenen bir düzeltme uygulanır:

$$h_{\text{render}} = h_{\text{WGS84}} + \delta z_{\text{surf}} \cdot \exp\!\left(-\left(\frac{\text{AGL}}{15.0\text{ m}}\right)^2\right)$$

Bu ağırlıklandırma sayesinde seyir irtifasında (`AGL > 45 m`, ağırlık ~0,0001) `LTFJ`/`LTBX` CTR duvarlarına göre mutlak jeodezik doğruluk bozulmazken, `AGL = 0,00 m` temas anında iniş takımları CesiumJS'te görünen çatı veya vertiport pedine sıfır boşlukla oturur.

### Referans Uygulama: `pyproj` (Python) ve `PROJ 9` (C++20) Modülleri

Aşağıdaki Python modülü Blender/ön-işleme hattında (`EPSG:9518` -> `EPSG:4979`, yerel `us_nga_egm08_25.tif` gridi veya `PROJ_NETWORK=ON` ile), C++20 başlığı ise `libvertirta` içinde iş parçacığı-güvenli `PJ_CONTEXT` ile doğrudan derlenebilir biçimde sunulmuştur:

```python
# vertirta_geodesy.py — Canonical Geodetic & Vertical Datum Bridge (EGM2008 <-> WGS84 HAE)
import math
import numpy as np
from pyproj import Transformer

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = 2.0 * WGS84_F - WGS84_F * WGS84_F

# Kartal Coastal Vertiport Anchor (40.8860 N, 29.1880 E, 12.0 m AMSL EGM2008)
ANCHOR_LAT_DEG = 40.8860
ANCHOR_LON_DEG = 29.1880
ANCHOR_H_AMSL = 12.0
R_GAUSS_KARTAL = 6375036.971  # sqrt(M_curv * N_curv) at 40.8860 N

# EPSG:9518 = WGS84 + EGM2008 height (EPSG:4326 + EPSG:3855) -> EPSG:4979 = WGS84 3D HAE
_T_EGM08_TO_WGS84 = Transformer.from_crs("EPSG:9518", "EPSG:4979", always_xy=True)

def geoid_undulation_egm2008(lon_deg: float, lat_deg: float) -> float:
    """Returns N_EGM2008(lat, lon) in meters (+36.6 to +36.8 m across Istanbul)."""
    _, _, h_hae_at_zero_amsl = _T_EGM08_TO_WGS84.transform(lon_deg, lat_deg, 0.0)
    return float(h_hae_at_zero_amsl)

def geodetic_to_ecef(lat_deg: float, lon_deg: float, h_hae: float) -> np.ndarray:
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    sL, cL = math.sin(lat), math.cos(lat)
    N_nu = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sL * sL)
    return np.array([
        (N_nu + h_hae) * cL * math.cos(lon),
        (N_nu + h_hae) * cL * math.sin(lon),
        (N_nu * (1.0 - WGS84_E2) + h_hae) * sL
    ], dtype=np.float64)

def ecef_to_geodetic(xyz: np.ndarray) -> tuple[float, float, float]:
    X, Y, Z = float(xyz[0]), float(xyz[1]), float(xyz[2])
    p = math.hypot(X, Y)
    lon = math.atan2(Y, X)
    lat = math.atan2(Z, p * (1.0 - WGS84_E2))
    for _ in range(8):
        sL = math.sin(lat)
        N_nu = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sL * sL)
        h = p / math.cos(lat) - N_nu
        lat = math.atan2(Z, p * (1.0 - WGS84_E2 * N_nu / (N_nu + h)))
    sL = math.sin(lat)
    N_nu = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sL * sL)
    h = p / math.cos(lat) - N_nu
    return math.degrees(lat), math.degrees(lon), h

def R_ecef_from_enu(lat_deg: float, lon_deg: float) -> np.ndarray:
    sL, cL = math.sin(math.radians(lat_deg)), math.cos(math.radians(lat_deg))
    sO, cO = math.sin(math.radians(lon_deg)), math.cos(math.radians(lon_deg))
    return np.array([
        [-sO, -sL * cO, cL * cO],
        [ cO, -sL * sO, cL * sO],
        [0.0,  cL,      sL     ]
    ], dtype=np.float64)

class VertiRTAGeodesyBridge:
    def __init__(self):
        self.N0 = geoid_undulation_egm2008(ANCHOR_LON_DEG, ANCHOR_LAT_DEG)
        self.h0 = ANCHOR_H_AMSL + self.N0
        self.p0_ecef = geodetic_to_ecef(ANCHOR_LAT_DEG, ANCHOR_LON_DEG, self.h0)
        self.R0 = R_ecef_from_enu(ANCHOR_LAT_DEG, ANCHOR_LON_DEG)

    def flat_enu_to_cesium_wgs84(self, E: float, N: float, Z_flat: float) -> dict:
        """Eliminates both +36.7 m geoid offset AND up to +8.82 m curvature float."""
        u_tan = Z_flat - (E * E + N * N) / (2.0 * R_GAUSS_KARTAL)
        p_corr = self.p0_ecef + self.R0 @ np.array([E, N, u_tan])
        lat_deg, lon_deg, _ = ecef_to_geodetic(p_corr)
        H_amsl = ANCHOR_H_AMSL + Z_flat
        N_local = geoid_undulation_egm2008(lon_deg, lat_deg)
        h_hae = H_amsl + N_local
        return {
            "lat_deg": lat_deg, "lon_deg": lon_deg,
            "alt_amsl_m": H_amsl, "geoid_n_m": N_local, "alt_hae_m": h_hae,
            "ecef_m": geodetic_to_ecef(lat_deg, lon_deg, h_hae)
        }

    @staticmethod
    def px4_quat_to_enu_flu(w: float, x: float, y: float, z: float) -> tuple[float, float, float, float]:
        """Exact closed-form q_ENU_FLU from PX4 q_NED_FRD (Hamilton [w, x, y, z])."""
        s = math.sqrt(0.5)
        return (s * (w + z), s * (x + y), s * (x - y), s * (w - z))
```

```cpp
// vertirta_geodesy.hpp — Portable C++20 Reference Transform for libvertirta (.vrta logger)
// Links against PROJ 9+ (libproj) with local us_nga_egm08_25.tif in PROJ_DATA
#pragma once
#include <array>
#include <cmath>
#include <stdexcept>
#include <proj.h>

namespace vertirta::geo {

constexpr double kPi           = 3.14159265358979323846;
constexpr double kDeg2Rad      = kPi / 180.0;
constexpr double kRad2Deg      = 180.0 / kPi;
constexpr double kWGS84_A      = 6378137.0;
constexpr double kWGS84_F      = 1.0 / 298.257223563;
constexpr double kWGS84_E2     = 2.0 * kWGS84_F - kWGS84_F * kWGS84_F;
constexpr double kAnchorLatDeg = 40.8860;
constexpr double kAnchorLonDeg = 29.1880;
constexpr double kAnchorHAmsl  = 12.0;
constexpr double kRGaussKartal = 6375036.971;

struct GeodeticPose {
    double lat_deg, lon_deg, alt_amsl_m, geoid_n_m, alt_hae_m;
    std::array<double, 3> ecef_m;
};

class GeodesyTransform {
public:
    GeodesyTransform() {
        ctx_ = proj_context_create();
        proj_context_set_enable_network(ctx_, 0); // Deterministic offline grid lookup
        PJ* raw = proj_create_crs_to_crs(ctx_, "EPSG:9518", "EPSG:4979", nullptr);
        if (!raw) throw std::runtime_error("PROJ failed to init EPSG:9518 -> EPSG:4979");
        pj_ = proj_normalize_for_visualization(ctx_, raw); // Enforce lon, lat, z order
        proj_destroy(raw);
        N0_ = QueryGeoidN(kAnchorLonDeg, kAnchorLatDeg);
        p0_ecef_ = GeodeticToEcef(kAnchorLatDeg, kAnchorLonDeg, kAnchorHAmsl + N0_);
        InitR0(kAnchorLatDeg, kAnchorLonDeg);
    }
    ~GeodesyTransform() {
        if (pj_)  proj_destroy(pj_);
        if (ctx_) proj_context_destroy(ctx_);
    }

    double QueryGeoidN(double lon_deg, double lat_deg) const {
        PJ_COORD in = proj_coord(lon_deg, lat_deg, 0.0, 0.0);
        PJ_COORD out = proj_trans(pj_, PJ_FWD, in);
        return out.xyz.z; // WGS84 HAE at 0.0 m EGM2008 AMSL == N_EGM2008
    }

    GeodeticPose FlatEnuToCesium(double E, double N, double Z_flat) const {
        const double u_tan = Z_flat - (E * E + N * N) / (2.0 * kRGaussKartal);
        std::array<double, 3> p_corr{
            p0_ecef_[0] + R0_[0]*E + R0_[1]*N + R0_[2]*u_tan,
            p0_ecef_[1] + R0_[3]*E + R0_[4]*N + R0_[5]*u_tan,
            p0_ecef_[2] + R0_[6]*E + R0_[7]*N + R0_[8]*u_tan
        };
        auto [lat, lon, _] = EcefToGeodetic(p_corr);
        const double H_amsl = kAnchorHAmsl + Z_flat;
        const double N_loc  = QueryGeoidN(lon, lat);
        const double h_hae  = H_amsl + N_loc;
        return {lat, lon, H_amsl, N_loc, h_hae, GeodeticToEcef(lat, lon, h_hae)};
    }

    static std::array<double, 4> Px4QuatToEnuFlu(double w, double x, double y, double z) {
        constexpr double s = 0.70710678118654752440;
        return {s * (w + z), s * (x + y), s * (x - y), s * (w - z)};
    }

private:
    static std::array<double, 3> GeodeticToEcef(double lat_deg, double lon_deg, double h) {
        const double lat = lat_deg * kDeg2Rad, lon = lon_deg * kDeg2Rad;
        const double sL = std::sin(lat), cL = std::cos(lat);
        const double N_nu = kWGS84_A / std::sqrt(1.0 - kWGS84_E2 * sL * sL);
        return {(N_nu + h) * cL * std::cos(lon),
                (N_nu + h) * cL * std::sin(lon),
                (N_nu * (1.0 - kWGS84_E2) + h) * sL};
    }

    static std::array<double, 3> EcefToGeodetic(const std::array<double, 3>& p) {
        const double r = std::hypot(p[0], p[1]);
        const double lon = std::atan2(p[1], p[0]);
        double lat = std::atan2(p[2], r * (1.0 - kWGS84_E2));
        double h = 0.0;
        for (int i = 0; i < 8; ++i) {
            const double sL = std::sin(lat);
            const double N_nu = kWGS84_A / std::sqrt(1.0 - kWGS84_E2 * sL * sL);
            h = r / std::cos(lat) - N_nu;
            lat = std::atan2(p[2], r * (1.0 - kWGS84_E2 * N_nu / (N_nu + h)));
        }
        return {lat * kRad2Deg, lon * kRad2Deg, h};
    }

    void InitR0(double lat_deg, double lon_deg) {
        const double sL = std::sin(lat_deg * kDeg2Rad), cL = std::cos(lat_deg * kDeg2Rad);
        const double sO = std::sin(lon_deg * kDeg2Rad), cO = std::cos(lon_deg * kDeg2Rad);
        R0_ = {-sO, -sL*cO, cL*cO,  cO, -sL*sO, cL*sO,  0.0, cL, sL};
    }

    PJ_CONTEXT* ctx_{nullptr};
    PJ* pj_{nullptr};
    double N0_{36.68};
    std::array<double, 3> p0_ecef_{};
    std::array<double, 9> R0_{};
};

} // namespace vertirta::geo
```

---

## 4. Uçtan Uca `.vrta` -> Colab G4 Başsız Blender Cycles OptiX İşleme Planı ve Canlı `:5180` Şeması (Teslimat 4)

### Aşama 1: %100 ToS-Temiz Açık Veri Hazırlama Hattı

Çevrimdışı İstanbul sahnesi, tek bir kabuk betiğiyle Copernicus GLO-30 DEM, Geofabrik OSM vektörleri ve Overture Maps bina katmanlarını yalnızca 15 x 15 km Kartal kutusuna kırparak oluşturur:

```bash
# 1. Copernicus GLO-30 DEM (Tek 1x1 derece COG karosu, kimlik doğrulamasız)
aws s3 cp --no-sign-request \
  s3://copernicus-dem-30m/Copernicus_DSM_COG_10_N40_00_E029_00_DEM/Copernicus_DSM_COG_10_N40_00_E029_00_DEM.tif \
  ./raw/cop30_N40E029.tif

# 2. Yerel UTM 35N (EPSG:32635) 15x15 km Kartal merkezli kırpma ve yeniden örnekleme
gdalwarp -t_srs EPSG:32635 -te 677019 4520725 691640 4536096 -tr 10 10 -r cubicspline \
  -dstnodata 0.0 ./raw/cop30_N40E029.tif ./stage/kartal_15km_egm2008.tif

# 3. Geofabrik OSM Türkiye vektörlerinden 15x15 km kırpma (osmium-tool)
wget -q [https://download.geofabrik.de/europe/turkey-latest.osm.pbf](https://download.geofabrik.de/europe/turkey-latest.osm.pbf) -O ./raw/turkey-latest.osm.pbf
osmium extract -b 29.099011,40.818464,29.276989,40.953536 \
  ./raw/turkey-latest.osm.pbf -o ./stage/kartal_15km.osm.pbf

# 4. Overture Maps Buildings (STAC üzerinden en güncel sürüm, DuckDB S3 GeoParquet sorgusu)
duckdb -c "
INSTALL spatial; LOAD spatial; INSTALL httpfs; LOAD httpfs;
SET s3_region='us-west-2';
SET VARIABLE latest = (SELECT latest FROM '[https://stac.overturemaps.org/catalog.json](https://stac.overturemaps.org/catalog.json)');
COPY (
  SELECT id, height, num_floors, roof_shape, roof_material, roof_color,
         facade_material, subtype, class, geometry
  FROM read_parquet('s3://overturemaps-us-west-2/release/' |

| getvariable('latest') |
| '/theme=buildings/type=building/*.parquet', hive_partitioning=1)
  WHERE bbox.xmin <= 29.276989 AND bbox.xmax >= 29.099011
    AND bbox.ymin <= 40.953536 AND bbox.ymax >= 40.818464
) TO './stage/kartal_15km_overture_buildings.parquet' (FORMAT PARQUET);
"
```

Sentinel-2 L2A `35TPF` sahnesi (B04/B03/B02 RGB + B08 NIR), `pystac-client` ile Element 84 Earth Search STAC API (`https://earth-search.aws.element84.com/v1`, `eo:cloud_cover < 2`) üzerinden çekilerek aynı `EPSG:32635` sınırına kırpılır; `(B08 - B04)/(B08 + B04)` ile **NDVI** (park/bitki örtüsü maskesi) ve `(B03 - B08)/(B03 + B08)` ile **NDWI** (Marmara kıyı maskesi) 16-bit gri tonlamalı GeoTIFF olarak Blender gölgelendiricisine aktarılır. Bina yükseklikleri üç kademeli deterministik öncelikle atanır: önce OSM/Overture `height` (m) alanı, bu boşsa `num_floors` veya `building:levels` x `3,15 m` + çatı payı (`1,8 m` kırma çatı / `1,1 m` düz parapet), her ikisi de yoksa İBB *Mahalle Bazlı Bina Analiz Verisi* (`CC BY 4.0`) kat dağılımından bina taban alanına ve deterministik GERS/OSM kimlik özetine (hash) bağlı kat sayısı türetilir.

### Aşama 2: Colab G4 Başsız Blender 5.2 LTS Kurulumu ve Kesin CLI Komutları

Blender komut satırı argümanları soldan sağa sırayla yürütüldüğünden `.blend` dosyası, motor (`-E CYCLES`), Python betiği (`-P`), çıktı şablonu (`-o`), kare aralığı (`-s`/`-e`), animasyon tetikleyicisi (`-a`) ve çift tire (`--`) ardındaki Cycles cihaz seçenekleri (`--cycles-device OPTIX --cycles-print-stats`) kesin sırayla verilmelidir. Kullanıcı raporlarında Blackwell GPU'larda uzun süreli sürekli yük altında nadir `Xid 79/120` sürücü sıfırlamaları bildirildiğinden ve Colab bağlantısı kopabileceğinden, 1.800 karelik (60 sn @ 30 fps) 4K işleme **150 karelik devam ettirilebilir dilimlerle** yürütülüp her dilim sonunda kalıcı depolamaya aktarılır:

```bash
# Colab G4 Kurulumu (Sürücü >= 575 için Blender 5.2 LTS; 535-574 aralığında 4.5.16 LTS kullanın)
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
wget -q [https://download.blender.org/release/Blender5.2/blender-5.2.4-linux-x64.tar.xz](https://download.blender.org/release/Blender5.2/blender-5.2.4-linux-x64.tar.xz)
tar -xf blender-5.2.4-linux-x64.tar.xz -C /opt/ && ln -sf /opt/blender-5.2.4-linux-x64/blender /usr/local/bin/blender

# OptiX aygıt ve libnvoptix doğrulaması (RTX PRO 6000 Blackwell OPTIX olarak listelenmelidir)
blender -b --python-expr "
import bpy
p = bpy.context.preferences.addons['cycles'].preferences
p.compute_device_type = 'OPTIX'; p.get_devices()
print([(d.name, d.type) for d in p.devices])
"

# 150 karelik dayanıklı dilimlerle 4K UHD 30 fps Cycles OptiX işleme
for START in $(seq 1 150 1800); do
  END=$((START + 149))
  [ $END -gt 1800 ] && END=1800
  blender -b /content/work/vertirta_istanbul_master.blend \
    -E CYCLES \
    -P /content/scripts/apply_vrta_and_render_setup.py \
    -o /content/renders_4k/frame_#### \
    -F PNG \
    -s $START -e $END \
    -a \
    -- --cycles-device OPTIX --cycles-print-stats \
    --vrta /content/logs/kartal_m3_fault.vrta \
    --res 3840x2160 --samples 128
  rsync -a /content/renders_4k/ /content/drive/MyDrive/VertiRTA_4K_Frames/
done
```

### Aşama 3: `bpy` OptiX + Persistent Data Yapılandırması ve Prosedürel Kıyı PBR Gölgelendirici Aşamaları

Aşağıdaki `bpy` modülü hem **Blender 5.2 LTS** hem de **4.5 LTS** ile uyumludur: `Principled BSDF v2` soket adlarını (`Specular IOR Level`, `Coat Weight`, `Emission Color`, `Emission Strength`, `Transmission Weight`) doğru kullanır, `mat.use_nodes` çağrısını 5.x uyarısından korur, yalnızca RTX PRO 6000 OptiX aygıtını açıp CPU'yu kapatır ve çok-kareli animasyonlarda sahne/BVH yeniden kurulum maliyetini bir render-farm satıcı testinde (`gpurenderstudio.com`, RTX 5090, 4 sahne 10 kare) ortalama **%37,3 (en fazla %57,3)** düşüren **`use_persistent_data = True`** bayrağını etkinleştirir:

```python
# apply_vrta_and_render_setup.py (OptiX + Persistent Data + Coastal Procedural PBR Stages)
import bpy

def configure_cycles_blackwell_optix(width=3840, height=2160, max_samples=128):
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'

    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'OPTIX'
    prefs.get_devices()
    for d in prefs.devices:
        d.use = (d.type == 'OPTIX')  # GPU-only; prevents CPU tile stalls

    scene.render.use_persistent_data = True
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.015
    scene.cycles.samples = max_samples
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = 'OPTIX'
    scene.cycles.denoising_input_passes = 'RGB_ALBEDO_NORMAL'

    scene.cycles.max_bounces = 6
    scene.cycles.diffuse_bounces = 3
    scene.cycles.glossy_bounces = 3
    scene.cycles.transmission_bounces = 4
    scene.cycles.transparent_max_bounces = 16  # Clean CTR curtains & rotor blur discs
    scene.cycles.volume_bounces = 1            # Failed-motor smoke plume

    # True 3D arc motion blur (set obj.cycles.motion_steps = 5 ONLY on coaxial rotors)
    scene.render.use_motion_blur = True
    scene.render.motion_blur_position = 'CENTER'
    scene.render.motion_blur_shutter = 0.50

    scene.render.resolution_x, scene.render.resolution_y = width, height
    scene.render.resolution_percentage = 100
    scene.render.fps = 30
    scene.view_settings.view_transform = 'AgX'
    scene.view_settings.look = 'AgX - Medium High Contrast'

def _new_material(name: str):
    mat = bpy.data.materials.new(name=name)
    if bpy.app.version < (5, 0, 0):
        mat.use_nodes = True  # Required on 4.5 LTS; deprecated on 5.x (auto-created)
    return mat

def build_turkish_rooftop_concrete_slz_shader():
    """1 m AGL photorealistic reinforced-concrete Kartal/Dragos rooftop PBR (100% procedural)."""
    mat = _new_material("Mat_Istanbul_Concrete_Rooftop_SLZ")
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links
    bsdf = nodes.get("Principled BSDF")

    tex_coord = nodes.new("ShaderNodeTexCoord")
    noise_macro = nodes.new("ShaderNodeTexNoise")
    noise_macro.inputs["Scale"].default_value = 0.35
    noise_macro.inputs["Detail"].default_value = 8.0
    noise_macro.inputs["Roughness"].default_value = 0.65

    noise_micro = nodes.new("ShaderNodeTexNoise")
    noise_micro.inputs["Scale"].default_value = 18.0
    noise_micro.inputs["Detail"].default_value = 10.0

    joints = nodes.new("ShaderNodeTexBrick")
    joints.inputs["Scale"].default_value = 0.45
    joints.inputs["Mortar Size"].default_value = 0.012
    joints.inputs["Mortar Smooth"].default_value = 0.2
    joints.inputs["Color1"].default_value = (0.34, 0.33, 0.31, 1.0)
    joints.inputs["Color2"].default_value = (0.28, 0.28, 0.27, 1.0)
    joints.inputs["Mortar"].default_value = (0.09, 0.09, 0.09, 1.0)

    mix_col = nodes.new("ShaderNodeMix")
    mix_col.data_type = 'RGBA'
    mix_col.blend_type = 'MULTIPLY'
    mix_col.inputs["Factor"].default_value = 0.32

    ramp_rough = nodes.new("ShaderNodeValToRGB")
    ramp_rough.color_ramp.elements[0].position = 0.25
    ramp_rough.color_ramp.elements[0].color = (0.68, 0.68, 0.68, 1.0)
    ramp_rough.color_ramp.elements[1].position = 0.80
    ramp_rough.color_ramp.elements[1].color = (0.94, 0.94, 0.94, 1.0)

    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.22
    bump.inputs["Distance"].default_value = 0.015

    links.new(tex_coord.outputs["Object"], noise_macro.inputs["Vector"])
    links.new(tex_coord.outputs["Object"], noise_micro.inputs["Vector"])
    links.new(tex_coord.outputs["Object"], joints.inputs["Vector"])
    links.new(joints.outputs["Color"], mix_col.inputs["A"])
    links.new(noise_macro.outputs["Fac"], mix_col.inputs["B"])
    links.new(mix_col.outputs["Result"], bsdf.inputs["Base Color"])
    links.new(noise_macro.outputs["Fac"], ramp_rough.inputs["Fac"])
    links.new(ramp_rough.outputs["Color"], bsdf.inputs["Roughness"])
    links.new(noise_micro.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    bsdf.inputs["Specular IOR Level"].default_value = 0.42
    bsdf.inputs["Coat Weight"].default_value = 0.0
    return mat

def build_marmara_ocean_shader():
    """Sea of Marmara specular water with dual-scale procedural swell and ripples."""
    mat = _new_material("Mat_Marmara_Sea")
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links
    bsdf = nodes.get("Principled BSDF")

    tex_coord = nodes.new("ShaderNodeTexCoord")
    swell = nodes.new("ShaderNodeTexWave")
    swell.wave_type = 'BANDS'
    swell.inputs["Scale"].default_value = 0.08
    swell.inputs["Distortion"].default_value = 4.5
    swell.inputs["Detail"].default_value = 5.0

    ripples = nodes.new("ShaderNodeTexNoise")
    ripples.inputs["Scale"].default_value = 1.8
    ripples.inputs["Detail"].default_value = 9.0
    ripples.inputs["Roughness"].default_value = 0.55

    mix_h = nodes.new("ShaderNodeMath")
    mix_h.operation = 'ADD'
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.14
    bump.inputs["Distance"].default_value = 0.08

    bsdf.inputs["Base Color"].default_value = (0.012, 0.044, 0.068, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.045
    bsdf.inputs["IOR"].default_value = 1.333
    bsdf.inputs["Specular IOR Level"].default_value = 0.55
    bsdf.inputs["Coat Weight"].default_value = 0.15

    links.new(tex_coord.outputs["Object"], swell.inputs["Vector"])
    links.new(tex_coord.outputs["Object"], ripples.inputs["Vector"])
    links.new(swell.outputs["Fac"], mix_h.inputs[0])
    links.new(ripples.outputs["Fac"], mix_h.inputs[1])
    links.new(mix_h.outputs["Value"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat
```

10 m çözünürlüklü Sentinel-2 L2A altlığının yakın mesafede pikselleşmesini tamamen ortadan kaldırmak için sahne üç ek prosedürel PBR aşamasıyla zenginleştirilir:

1. **Mesafe-Tabanlı Arazi ve Kıyı Karıştırma (`Camera Data -> View Z Depth`):** 250 m ötede Sentinel-2 L2A renk tonu baskın kalırken 25–250 m arasında pürüzsüz geçişle OSM yol vektörlerine prosedürel asfalt ve şerit çizgisi, Kartal/Maltepe sahil bandına beton/traverten yürüyüş yolu ve NDVI maskeli Dragos/Maltepe parklarına Geometry Nodes çim ile çam ağacı örnekleri (instancing) devreye girer.
2. **Türk Kıyı Mimarisi Çatı ve Cephe Üreticisi (Geometry Nodes):** Düşük ve orta katlı Dragos/Maltepe konutlarına prosedürel dalgalı **terakota kiremit** çatı (`Roughness 0,68–0,85`, AO toz/yosun karartmalı), yüksek katlı Kartal yapıları ile aday SLZ çatılarına yukarıdaki betonarme-zift ek yeri gölgelendiricisi üzerine parapet duvarı, klima dış ünitesi, su deposu, asansör kulesi ve Kartal Vertiportu'nda standart sarı/beyaz `H` TLOF/FATO işaretlemesi yerleştirilir; cepheler dünya-Z ekseninde 3,15 m kat modülüyle sıva (`Roughness 0,85`) ve cam pencere (`IOR 1,50`, `Roughness 0,04`) şeritleri oluşturur.
3. **Fiziksel Atmosfer (`ShaderNodeTexSky — Nishita`):** İstanbul enlem-boylamına (`40,886° K, 29,188° D`) kilitli geç öğleden sonra Marmara çapraz güneşi sıfır HDRI VRAM maliyetiyle doğal kıyı sisi ve atmosferik derinlik sağlar.

### Aşama 4: Uçuş Testi Havacılık Katmanları ve 60 Saniyelik Kamera Kurgusu

Tüm dinamik görsel katmanlar doğrudan `.vrta` ikili zaman serisinden sürülür (fizik ve HJ CBVF türetimi kapsam dışı tutulup önceden hesaplanmış zaman-endeksli durum vektörleri ve isoyüzey halkaları okunur).

#### Yarı Saydam 3B `LTFJ` ve `LTBX` CTR Hava Sahası Perdeleri

DHMİ eAIP'den üretilen 5 NM (`9.260 m`) dairesel yaylar, `40°59'15"K, 029°19'30"D` ile `40°54'46,03"K, 029°11'31,64"D` ortak kesim doğrusu boyunca kırpılarak kapalı 3B hava sahası duvarlarına dönüştürülür. `LTFJ` perdesi kırmızı (`SFC – 762,0 m AMSL`), `LTBX` perdesi kehribar (`SFC – 609,6 m AMSL`) renkli olup her `500 FT` (`152,4; 304,8; 457,2; 609,6; 762,0 m AMSL`) seviyesinde parlak irtifa çizgileri taşır. Blender Cycles'ta kenarları parlak, ortası %12 opak holografik `Layer Weight (Facing) -> Mix Shader (Transparent BSDF + Emission)` malzemesi kullanılır ve perdelerin İstanbul binalarına yapay gölge düşürmemesi için `obj.visible_shadow = False` ile `obj.visible_diffuse = False` zorunlu olarak ayarlanır. CesiumJS'te aynı geometri `Cesium.WallGraphics` (`maximumHeights`: `798,58 m` / `646,35 m` WGS84 HAE) + 500 FT `PolylineGraphics` halkalarıyla çizilir ve `viewer.scene.globe.depthTestAgainstTerrain = true` etkinleştirilerek duvarların Dragos Tepesi arkasından görünmesi engellenir.

#### Arıza-Bozulmalı 4B CBVF Güvenlik Hunisi ve Daralan İniş Konisi

`.vrta` akışından okunan 24 köşeli erişilebilirlik halkaları, Blender'da her karede araç konumundan arazi ve bina yüzeyine uzanan dinamik bir 3B kabuk mesh'ine güncellenir. Nominal 8-rotor uçuşunda camgöbeği (`#12B5CB`) görünen huni, `.vrta` motor arızası anında (`t == t_fault`) hızla daralıp kehribar/kırmızı tona geçer; eşzamanlı olarak arazi ve bina koleksiyonuna `Shrinkwrap (PROJECT, -Z)` ile projekte edilen yüksek kontrastlı zemin/çatı halkası, daralan iniş konisinin içinde kalan güvenli acil iniş alanını (SLZ) net biçimde işaretler. CesiumJS'te zemin projeksiyonu `CallbackProperty` ile güncellenen ve `classificationType: Cesium.ClassificationType.BOTH` ayarıyla hem arazi hem OSM bina çatıları üzerine kusursuz giydirilen `PolygonGraphics` ile verilir.

#### Koaksiyel Rotor Hareket Bulanıklığı, İtki Çubukları ve Arızalı Motor Dumanı

30 fps'de ~2.400 RPM dönen bir rotor tek kare süresinde 1,33 tam tur attığından tek adımlı doğrusal hareket bulanıklığı palleri kord çizgisine çökertir. Bunu aşmak için üç rejimli hibrit yöntem uygulanır: nominal devirde (`RPM > 1.200`) fiziksel pal gizlenip `.vrta` `rotor_rpm[i]` değerine bağlı anizotropik yarı saydam **rotor bulanıklık diski** gösterilir; motor arızası tetiklenip devir düşerken (`0 < RPM <= 1.200`) fiziksel 3B pal ağı açılır ve yalnızca 8 rotor nesnesinde `rotor_obj.cycles.motion_steps = 5` ($2^{5-1} = 16$ kavisli yay alt adımı) ile gerçek dairesel yay bulanıklığı işlenir; `RPM == 0` anında arızalı koaksiyel pal sabitlenmiş biçimde net görünür. Her rotor göbeğindeki emisyon itki vektör çubuğu `.vrta` itki değerine göre ölçeklenip renklenir (yeşil: nominal, parlak camgöbeği: telafi aşırı itkisi, kırmızı/sönük: `0 N` arızalı motor). Kesin `.vrta` arıza karesinde (`active_motor_mask` bit düşümü) arızalı naselden başlayan **Geometry Nodes Simulation Zone -> Points to Volume -> Set Material (`Principled Volume`)** hattı uçak hız vektörü ve aşağı akışla sürüklenen elektrik dumanı üretir ve çoklu-dilim kare tutarlılığı için işleme öncesinde diske **bake** edilir. (Hızlı önizleme işlerinde `transparent_max_bounces = 32` ile duman kartları alternatif olarak kullanılabilir.) CesiumJS'te ise tek `VertiRTA-X8-240.glb` içindeki 8 rotor düğümü `Cesium.NodeTransformationProperty` ile RPM'den döndürülüp arıza anında durdurulur ve duman için `Cesium.ParticleSystem` araca bağlanır.

#### 60 Saniyelik (1.800 Kare @ 30 fps) Kamera Kurgusu ve `ffmpeg` NVENC HUD Birleştirme

| Kare Aralığı (Süre) | Kamera Açısı / Odak Uzaklığı | Anlatım Odağı ve Uçuş Olayı |
| :--- | :--- | :--- |
| **1–360 (0–12 sn)** | Marmara Kıyı Genel Planı (`35 mm`) | Marmara Denizi üzerinden KD'ya bakış; Kartal Vertiportu kalkışı, arka planda Dragos Tepesi, Maltepe sahili ve 1 km doğuda kırmızı `LTFJ CTR` duvarı. |
| **361–840 (12–28 sn)** | Dinamik Takip Kamerası (`Chase Cam, 50 mm, 3/4 Arka-Yan`) | Seyir fazı ve kesin `.vrta` arıza anı: arızalı koaksiyel palin durması, duman çıkışı, komşu rotorlarda aşırı itki rengi ve daralan CBVF güvenlik hunisi. |
| **841–1320 (28–44 sn)** | Kokpit HUD ve Nadir SLZ Karın Kamerası (`24 mm`) | Burun bakışından aşağı bakan karın kamerasına geçiş; Dragos/Kartal çatı dokusu ve daralan erişilebilirlik halkasının hedef SLZ çatısına kilitlenmesi. |
| **1321–1800 (44–60 sn)** | Çatı İniş Pedi Telefoto Kamerası (`85 mm`) | Hedef SLZ çatısında 1,5 m göz hizasından pan takip; prosedürel beton/kiremit yakın planı ve tam `0,00 m AGL` iniş takımı teması. |

Ham Blender PNG kareleri (`frame_####.png`) ile Python (`Pillow`/`pycairo`) tarafından `.vrta` kaydından kare-kare üretilen şeffaf RGBA telemetri HUD dizisi (`hud_####.png` — `V`, `AGL`, `SoC`, `V_term`, `TMR Status`, `CBF Intervention ||u_safe - u_nom||`, 8-rotor `Active Motor Mask` diyagramı ve `LTFJ/LTBX` mesafe göstergesi), RTX PRO 6000 Blackwell'in **4x 9. Nesil NVENC** donanım kodlayıcısıyla (`hevc_nvenc -cq 18 -preset slow`) saniyeler içinde 4K 30 fps MP4 dosyasına birleştirilir:

```bash
ffmpeg -y \
  -framerate 30 -i /content/renders_4k/frame_%04d.png \
  -framerate 30 -i /content/hud_4k/hud_%04d.png \
  -filter_complex "[0:v][1:v]overlay=0:0:shortest=1,format=yuv420p" \
  -c:v hevc_nvenc -preset slow -rc vbr -cq 18 -b:v 0 -maxrate 65M -bufsize 130M \
  -tag:v hvc1 -movflags +faststart \
  /content/out/VertiRTA_X8_240_Istanbul_4K30_FlightReplay.mp4
```

### Colab G4 (RTX PRO 6000 Blackwell 96 GB VRAM) Bellek ve Kare-Süresi Bütçesi

NVIDIA RTX PRO 6000 Blackwell Server Edition; **96 GB GDDR7 ECC** bellek, **1.597 GB/s** bant genişliği, **24.064 CUDA çekirdeği** ve **188 adet 4. nesil RT çekirdeği** taşır. Üçüncü taraf Blender Open Data toplayıcılarında (Kasım 2026) aynı silikon çekirdek sayısına sahip RTX PRO 6000 Blackwell Workstation Edition, Blender 5.2.0 Cycles OptiX testlerinde **17.306** (19 koşu, `sourceit.com.sg`) ila **18.024** (`geturna.com`) medyan puanıyla RTX 5090'ın (16.080–16.171) yaklaşık %8–11, RTX 4090'ın (11.534–11.553) yaklaşık %50–56 önündedir (Server Edition için doğrudan Blender Open Data medyanı yayımlanmadığından bu projeksiyonlar 400–600 W ayarlanabilir TDP profiline bağlı mühendislik modelleme tahminleridir; V-Ray kıyaslamasında Server Edition Workstation'ı geçmiştir). Aşağıdaki VRAM ve 60 sn (1.800 kare) süre/maliyet bütçesi, Cycles geometri/BVH bellek oranları, `gpurenderstudio.com` RTX 5090 çok-kareli `Persistent Data` testleri, resmi Google Cloud `g4-standard-48` Spot fiyatlandırması (`$1,695/saat`) ve Google'ın resmi tablo yayımlamaması nedeniyle bağımsız kullanıcı ölçümlerine dayanan Colab Pro+ G4 birim yakım tahmini (`~11 saat / 100 birim`, `~$0,88/saat`, yalnızca Pro+) esas alınarak hesaplanmıştır:

| Kaynak / Çıktı Kalemi | Tahmini Değer / Aralık | 96 GB Blackwell Kapasite Oranı ve Maliyet Notu |
| :--- | :--- | :--- |
| **Cycles OptiX Çalışma Zamanı + CUDA/RT Çekirdekleri** | **1,2 GB** | Sabit çekirdek bellek yükü. |
| **Arazi Ağı (`15 x 15 km` GLO-30 + 2 m Yakın Alan Alt Bölme)** | **1,8 GB** | Blender 5.x uyarlamalı alt bölme. |
| **Kentsel Binalar (~95 bin OSM/Overture + GeoNodes ~25M Üçgen)** | **3,6 GB** | ~1 GB / 8M üçgen geometri tabanı. |
| **Kalıcı OptiX BVH Hızlandırma Yapısı (`Persistent Data`)** | **5,2 GB** | Geometri belleğinin ~1,5–2x BVH yükü; 1.800 kare boyunca yeniden kurulmaz. |
| **`VertiRTA-X8-240` CAD + Koaksiyel Rotorlar + CTR/CBVF Ağları** | **0,4 GB** | ~1,5M üçgen yüksek detaylı eVTOL ve hava sahası/CBVF katmanları. |
| **Prosedürel/Detay PBR Dokular + Sentinel-2 + 4K AOV/Denoiser** | **1,6 GB** | 4K `RGB + Albedo + Normal` OptiX Denoiser tamponları dahil. |
| **Geometry Nodes Duman Hacim Izgarası + Kare Önbelleği** | **0,7 GB** | Bake edilmiş Volume Grid + kare çalışma tamponu. |
| **TOPLAM TEPE VRAM TÜKETİMİ (Model Tahmini)** | **14,5 GB / 96,0 GB** | **%15,1 kullanım (`81,5 GB` boş başlık payı — sıfır OOM riski)**. |
| **60 sn @ 30 fps — 1080p (`1920x1080`, 96 spp, `AT = 0,015`)** | **1,8–3,0 sn/kare (med. 2,4 sn)** -> **0,90–1,50 sa (med. 1,20 sa)** | ~10,9 Colab birimi (**~$1,06** tahmini Colab Pro+ / **~$2,04** GCP Spot). |
| **60 sn @ 30 fps — 4K UHD (`3840x2160`, 128 spp, `AT = 0,015`)** | **6,0–10,5 sn/kare (med. 8,0 sn)** -> **3,00–5,25 sa (med. 4,00 sa)** | ~36,4 Colab birimi (**~$3,52** tahmini Colab Pro+ / **~$6,80** GCP Spot; 24 sa Pro+ limitinin çok altında). |

> **Şekil 4: 15 x 15 km İstanbul 4K Sinema Sahnesi 96 GB Blackwell VRAM'in Yalnızca %15,1'ini Kullanır ve 4 Saatte Biter**
> Üst kartlar 60 saniyelik (1.800 kare, 30 fps) uçuş tekrarı için toplam bellek ve süre/maliyet hedeflerini özetler. Orta panel, **Persistent Data** açıkken **14,5 GB** tepe VRAM dağılımını gösterir (**81,5 GB** boş rezerv). Alt panel, süre tahminlerinin dayandığı Blender 5.2.0 Open Data Cycles OptiX puanlarını (RTX PRO 6000 Blackwell = **17.306**) karşılaştırır.

### Aşama 5: CesiumJS (`:5180`) Canlı WebSocket Telemetri Şeması ve GTX 1650 (4 GB VRAM) Kilidi

Yerel Windows iş istasyonundaki `libvertirta` akış sunucusu (veya `.vrta` canlı tekrar oynatıcısı), `ws://127.0.0.1:5180/telemetry` üzerinden **30 Hz** hızında JSON (veya isteğe bağlı MessagePack) telemetri paketleri yayınlar; 24 köşeli daralan CBVF erişilebilirlik poligonu yalnızca **5 Hz** frekansında veya `update_seq` değişiminde gönderilir:

```json
{
  "schema_ver": "vrta.ws.v1.2",
  "seq": 14820,
  "sim_time_s": 24.700,
  "playback_mode": "LIVE_REPLAY",
  "geodetic": {
    "lat_deg": 40.908412,
    "lon_deg": 29.152384,
    "alt_amsl_m": 168.42,
    "geoid_n_egm2008_m": 36.66,
    "alt_hae_wgs84_m": 205.08,
    "agl_m": 60.42,
    "q_enu_flu": [-0.6921, -0.1410, 0.0884, -0.7024],
    "vel_enu_mps": [-14.20, 8.65, -2.15]
  },
  "propulsion": {
    "active_motor_mask": 251,
    "failed_motor_indices": [2],
    "rotor_rpm": [2480, 2510, 0, 2940, 2610, 2590, 2495, 2505],
    "rotor_thrust_n": [335.0, 342.5, 0.0, 448.0, 368.0, 362.0, 338.0, 340.5],
    "smoke_active": true
  },
  "safety_avionics": {
    "airspeed_mps": 16.76,
    "soc_pct": 61.4,
    "v_term_v": 412.8,
    "tmr_status": "3/3_NOMINAL",
    "cbf_delta_u_norm": 0.428,
    "active_slz_id": "SLZ_DRAGOS_ROOF_04",
    "dist_ltfj_ctr_m": 3890.5,
    "dist_ltbx_ctr_m": 1940.2
  },
  "cbvf_overlay": {
    "update_seq": 247,
    "status": "DEGRADED_7_ROTOR",
    "cone_apex_hae_m": 205.08,
    "footprint_llh_hae": [
      [29.1492, 40.9101, 144.6],
      [29.1541, 40.9098, 141.2]
    ]
  }
}
```

Varsayılan CesiumJS bellek bütçesi (`cacheBytes = 512 MB` + `512 MB` taşma + yüksek MSAA + geniş arazi önbelleği) 4 GB VRAM'li GTX 1650 üzerinde Windows DWM ile birlikte VRAM taşmasına ve doku titremesine yol açabilir. Aşağıdaki yapılandırma bellek sınırlarını doğrudan çözümlenen `Cesium3DTileset` örneğine uygulayarak (`cacheBytes`, `maximumCacheOverflowBytes`, `maximumScreenSpaceError`) GPU belleğini `< 1,8 GB` hedefinde sabitler:

```javascript
// cesium_live_ops_5180.js — GTX 1650 (4 GB VRAM) Hardened Live-Ops Configuration
const viewer = new Cesium.Viewer("cesiumContainer", {
  terrain: Cesium.Terrain.fromWorldTerrain({ requestVertexNormals: true }),
  // Live streaming ONLY: pass ArcGIS Location Platform API token for commercial use,
  // or swap to Cesium.IonImageryProvider.fromAssetId(2) (Bing) / Azure Maps
  baseLayer: Cesium.ImageryLayer.fromProviderAsync(
    Cesium.ArcGisMapServerImageryProvider.fromUrl(
      "[https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer](https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer)"
    )
  ),
  shadows: false,                    // Avoids heavy shadow maps on 4 GB VRAM
  msaaSamples: 2,                    // Crisp CTR/HUD edges at half 4x MSAA footprint
  requestRenderMode: true,           // Render on 30 Hz WS telemetry frame or camera input
  maximumRenderTimeChange: Infinity,
  creditContainer: "cesium-credits"  // Mandatory Esri / Cesium / OSM attribution line
});

viewer.scene.globe.depthTestAgainstTerrain = true;  // Occludes CTR walls behind Dragos Hill
viewer.scene.globe.tileCacheSize = 64;              // Constrained terrain tile cache
viewer.scene.globe.maximumScreenSpaceError = 2.5;

const osmBuildings = await Cesium.createOsmBuildingsAsync();
osmBuildings.cacheBytes = 256 * 1024 * 1024;                // 256 MB hard GPU cache target
osmBuildings.maximumCacheOverflowBytes = 128 * 1024 * 1024; // 128 MB overflow ceiling
osmBuildings.maximumScreenSpaceError = 24;                  // Balanced LOD for GTX 1650
osmBuildings.dynamicScreenSpaceError = true;
osmBuildings.dynamicScreenSpaceErrorDensity = 0.00278;
osmBuildings.dynamicScreenSpaceErrorFactor = 4.0;
osmBuildings.skipLevelOfDetail = true;
osmBuildings.foveatedScreenSpaceError = true;
viewer.scene.primitives.add(osmBuildings);
```

### Uygulama Kabul Kapıları (Doğrulama Kontrol Listesi)

1. **Hukuki ve ToS Denetimi:** Çevrimdışı `.blend` paketinde ve Colab G4 dizininde hiçbir Bing, Esri, Google veya Cesium ion varlığının bulunmadığı; Copernicus (`© DLR/Airbus/ESA`), Sentinel-2 (`Contains modified Copernicus Sentinel data`), OSM/Overture (`© OpenStreetMap contributors`) ve canlı akış atıf çubuğunun (`#cesium-credits`) eksiksiz korunduğu teyit edilmelidir.
2. **Sıfır Geoit ve Eğrilik Doğrulaması:** Kartal Vertiportu (`0 km`), Dragos SLZ (`4,60 km`) ve Maltepe Sahil SLZ (`6,75 km`) noktalarına yerleştirilen üç kalibrasyon küpü Gazebo Harmonic, Blender Cycles ve CesiumJS'te karşılaştırılarak düşey sapmanın `< 0,05 m` (hedef `0,00 m` çatı/ped teması) olduğu onaylanmalıdır.
3. **Arıza Kare Senkronizasyonu:** `.vrta` içindeki kesin `t_fault` karesinde Blender Cycles'ta 3 numaralı koaksiyel rotorun bulanıklık diskinden yavaşlayan fiziksel pale (`motion_steps = 5`) geçtiği, duman hacminin tetiklendiği, komşu rotor itki çubuğunun telafi rengine döndüğü ve `ffmpeg` HUD katmanında `Active Motor Mask` ile `CBF Intervention ||u_safe - u_nom||` değerlerinin aynı kare numarasında değiştiği tek-kare denetimiyle doğrulanmalıdır.
4. **Blackwell Performans ve Bellek Kapısı:** 1.800 karelik 4K toplu işleme başlatılmadan önce üç referans kare (genel kıyı planı, arıza/duman karesi, 1 m AGL çatı teması) `--cycles-print-stats` ile çalıştırılarak tepe VRAM'in `< 18 GB` ve kare süresinin `< 10,5 sn` olduğu ölçülmelidir.