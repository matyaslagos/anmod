## Szorgalmi a 3-ik analógiás nyelvmodelles házihoz

### A feladat
Egy alap lineárisan interpolált bigram nyelvmodell egy `w1 w2` bigram feltételes valószínűségét úgy becsüli meg, hogy kiszámolja

- a `w1 w2` bigram empirikus feltételes valószínűségét és
- a `w2` unigram empirikus valószínűségét[^1],

és ennek a két értéknek a súlyozott átlagát veszi, valamilyen általunk kiválasztott súlyokkal. Így ha a `w1 w2` bigram nem fordult elő a tanítóadatban, a modell akkor is tud neki nullánál nagyobb valószínűséget becsülni – és minél gyakoribb a `w2` szó, annál nagyobb valószínűséget kap a `w1 w2` bigram.

De lehet hogy egy ennél jobb módszer lenne a valószínűségek becslésére az, ha nem a `w2` szó _tokengyakorisága_ számítana (azaz az hogy összesen hányszor fordult elő egy bigram második szavaként), hanem a `w2` szó _típusgyakorisága_, azaz az hogy hány _különböző_ bigramban fordult elő második szóként (más szóval hogy hány különböző szó után fordult elő).

A szorgalmi feladat (amivel ki lehet váltani a három sima házi feladatot) az, hogy a `h3.py` fájlban az `itp_model()` függvényt írd át úgy hogy a modell a bigramok valószínűségeit a második szavak típusgyakoriságai alapján becsülje meg, és nézd meg hogy így jobb modellt kapunk-e. Fontos hogy ehhez ki kell találni azt is hogy hogyan _normalizáld_ a szavak típusgyakoriságait: ahhoz hogy tényleg valószínűségeket rendeljen a modell a bigramokhoz, meg kell oldani hogy az összes szó "típusvalószínűségeinek" az összege 1 legyen. Ezért valamivel el kell osztani a szavak típusgyakoriságait úgy hogy ez a feltétel teljesüljön, hasonlóan mint ahogy a szavak tokengyakoriságát elosztjuk a tanítóadat méretével ahhoz hogy megkapjuk a "tokenvalószínűségüket" (más néven az empirikus valószínűségüket). (Nem gond ha ezt nem sikerül kitalálni, ezt meg lehet nézni a házi pdf-ének a segítségében.)

[^1]: Itt nem azt nézzük hogy egy bigram _első_ szavaként hányszor fordult elő `w2`, hanem azt hogy egy bigram _második_ szavaként hányszor fordult elő, mert a `</s>` mondatzáró szimbólumra végződő bigramok valószínűségeit csak így tudjuk megbecsülni.

### Az `itp_model()` függvény leírása

Az `itp_model()` függvény nem csinál mást mint hogy kiszámolja a tanítóadatban előforduló bigramok empirikus feltételes valószínűségeit és az unigramok empirikus valószínűségeit, és feljegyzi ezeket egy szótárba. Ezt a függvényt kellene megváltoztatni úgy, hogy az unigramoknak ne az empirikus valószínűségeit (azaz normalizált tokengyakoriságait) jegyezze fel, hanem a normalizált típusgyakoriságait. Ehhez kommentekben jelöltem hogy melyik sorokat kell átírni:

- a 90. és 91. sorokat (ahol a normalizáló konstanst kapjuk meg, az eredeti modell esetében a bigramok tokengyakoriságainak az összegét), és
- a 97. sort (ahol az eredeti modell esetében az unigram tokengyakoriságát kapjuk meg – itt az új modellben a típusgyakoriságát kellene megkapnunk).

A megváltoztatott változatot hasznos új függvényként definiálni, pl. `itp_type_model()` néven, hogy könnyen össze lehessen hasonlítani a két modellt.

(A 73. sor azért a `defaultdict()` függvénnyel inicializálja a `prob_dict()` szótárt, mert így majd a tesztelés közben amikor egy olyan bigram feltételes valószínűségét próbáljuk megkapni ami nem fordult elő a tanítóadatban, akkor automatikusan 0-t kapunk (ha ezt egy sima `dict()` típusú szótárban próbálnánk ugyanígy megnézni akkor hibaüzenetet kapnánk) – de amúgy ugyanúgy működik mint a sima szótárak.)

### A h3.py-ban lévő függvények használata
Először töltsük le és helyezzük egy mappába a `h3.py` és a `grimm_full.txt` fájlokat és navigáljunk ebbe a mappába a parancssorban (vagy bármilyen más programozó felületen).

Importáljuk a `h3` modult:
```
import h3
```
Importáljuk a Grimm korpuszt szavak listáinak listájaként:
```
corpus = h3.txt_import('grimm_full.txt')
```
Osszuk szét véletlenszerűen a korpuszt tanítóadatra és tesztadatra (kb. 90%-a lesz tanítóadat és 10%-a tesztadat):
```
training_data, test_data = h3.train_test(corpus)
```
Építsük meg a modellünket a tanítóadat alapján:
```
model = h3.itp_model(training_data)
```
Mérjük meg a modell perplexitását a tesztadaton, általunk választott súlyozással:
```
h3.perplexity(test_data, model, 0.75, 0.25)
```
(Itt a bigramvalószínűségeket súlyozzuk 0.75-tel és az unigramvalószínűségeket 0.25-tel. Én ezeket a súlyokat találtam a legjobbnak, a modell perplexitása ilyenkor 90 és 95 között lesz.)
