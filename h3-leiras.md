## Szorgalmi a 3-ik analógiás nyelvmodelles házihoz

### A feladat
Egy alap lineárisan interpolált bigram nyelvmodell egy `w1 w2` bigram feltételes valószínűségét úgy becsüli meg, hogy kiszámolja

- a `w1 w2` bigram empirikus feltételes valószínűségét és
- a `w2` unigram empirikus valószínűségét,

és ennek a két értéknek a súlyozott átlagát veszi, valamilyen általunk kiválasztott súlyokkal. Így ha a `w1 w2` bigram nem fordult elő a tanítóadatban, a modell akkor is tud neki nullánál nagyobb valószínűséget becsülni – és minél gyakoribb a `w2` szó, annál nagyobb valószínűséget kap a `w1 w2` bigram.

De lehet hogy egy ennél jobb módszer lenne a valószínűségek becslésére az, ha nem a `w2` szó _tokengyakorisága_ számítana (azaz az hogy összesen hányszor fordult elő egy bigram második szavaként[^1]), hanem a `w2` szó _típusgyakorisága_, azaz az hogy hány _különböző_ bigramban fordult elő második szóként (más szóval hogy hány különböző szó után fordult elő).

A szorgalmi feladat (amivel ki lehet váltani a három sima házi feladatot) az, hogy a [h3.py](h3.py) fájlban az `itp_model()` függvényt írd át úgy hogy a modell a bigramok valószínűségeit a második szavak tokengyakoriságai helyett a típusgyakoriságai alapján becsülje meg, és nézd meg hogy így jobb modellt kapunk-e (lent a "h3.py-ban lévő függvények használata" szekció leírja hogy ezt hogyan lehet megnézni).

Fontos hogy ehhez ki kell találni azt is hogy hogyan _normalizáld_ a szavak típusgyakoriságait: ahhoz hogy tényleg valószínűségeket rendeljen a modell a bigramokhoz, meg kell oldani hogy az összes szó "típusvalószínűségeinek" az összege 1 legyen. Ezért valamivel el kell osztani a szavak típusgyakoriságait úgy hogy ez a feltétel teljesüljön, hasonlóan mint ahogy a szavak tokengyakoriságát elosztjuk a tanítóadat méretével ahhoz hogy megkapjuk a "tokenvalószínűségüket" (más néven az empirikus valószínűségüket). (Nem gond ha ezt nem sikerül kitalálni, ezt meg lehet nézni a házi pdf-ének a segítségében.)

[^1]: Itt nem azt nézzük hogy egy bigram _első_ szavaként hányszor fordult elő `w2`, hanem azt hogy egy bigram _második_ szavaként hányszor fordult elő, mert a `</s>` mondatzáró szimbólumra végződő bigramok valószínűségeit csak így tudjuk megbecsülni.

### A függvények leírása

A `txt_import()` függvény stringek listáinak listájaként importál egy txt fájlt, tehát a korpuszunk mondatai mind egy-egy stringlistának fognak megfelelni, pl. így: `['i', 'am', 'reading', 'this']`. Ezt a mondatlistát a `train_test()` függvény véletlenszerűen kettészedi 90% tanítóadatra és 10% tesztadatra.

Az `itp_model()` függvény a modell létrehozásához először összegyűjti a `freq_dict` szótárba a `frequencies()` függvénnyel azt hogy:

- a tanítóadatban a kontextusok után milyen szavak fordultak elő hányszor, és hogy
- a tanítóadatban a szavak előtt milyen kontextusok fordultak elő hányszor.

(Ez az utóbbi hasznos lesz a modell megváltoztatásához.) Így pl.:
- ha a `freq_dict['fw']['egy']['vízesést']` érték 4 (lent pirossal jelölve), az azt jelenti hogy a `'vízesést'` szó 4-szer fordult elő az `'egy'` kontextus után, és
- ha a `freq_dict['bw']['reggelt']['jó']` érték 10 (lent kékkel jelölve), az azt jelenti hogy a `'jó'` kontextus 10-szer fordult elő a `'reggelt'` szó előtt.

![freq_dict](https://github.com/matyaslagos/anmod/assets/47662384/b96ba226-10c4-4d86-8c77-8005019fe5f1)

Az `itp_model()` függvény ezután a `freq_dict` szótár adatai alapján kiszámolja és összegyűjti a `prob_dict` szótárba:[^2]
- a tanítóadatban előfordult bigramok empirikus feltételes valószínűségeit (úgy hogy végigmegy a `freq_dict['fw']` szótárban szereplő összes kontextuson) és
- a tanítóadatban előfordult unigramok empirikus valószínűségeit (úgy hogy végigmegy a `freq_dict['bw']` szótárban szereplő összes szón – ezt azért így csinálja mert így könnyebb lesz átírni hogy ne a szavak tokengyakoriságát nézze a modell hanem a fent meghatározott típusgyakoriságaikat).

[^2]: A 73. sor azért a `defaultdict()` függvénnyel inicializálja a `prob_dict()` szótárt, mert így majd a tesztelés közben amikor egy olyan bigram feltételes valószínűségét próbáljuk megkapni ami nem fordult elő a tanítóadatban, akkor automatikusan 0-t kapunk (ha ezt egy sima `dict()` típusú szótárban próbálnánk ugyanígy megnézni akkor hibaüzenetet kapnánk) – de amúgy ugyanúgy működik mint a sima szótárak.

Az így kapott `prob_dict` szótárban a kulcsok az unigramok (stringek) és bigramok (két stringet tartalmazó tuple-ök), az értékeik pedig az empirikus (feltételes) valószínűségeik. Tehát pl.
- ha a `prob_dict[('old', 'king')]` értéke 0.075675, az azt jelenti hogy a `'king'` szó 0.075675 valószínűséggel következett az `'old'` kontextus után a tanítóadatban, és
- ha a `prob_dict['king']` értéke 0.002987, az azt jelenti hogy a `'king'` szó empirikus valószínűsége 0.002987.

Végül a `perplexity()` függvény kiszámítja az interpolált modell perplexitását (ezt a következő órán fogjuk venni, ez azt méri hogy mennyire lepődik meg a modell tesztadaton, azaz minél kisebb annál jobb) az általunk megadott `bi_wt`-el súlyozva a bigramvalószínűségeket és `un_wt`-el súlyozva az unigramvalószínűségeket – fontos hogy a `bi_wt` és az `un_wt` számok összege 1 legyen (nekem a 0.75 bigram- és a 0.25 unigram-súly jött be a legjobban).

### Az `itp_model()` függvény megváltoztatása

Az `itp_model()` függvényt kellene megváltoztatni úgy, hogy az unigramoknak ne az empirikus valószínűségeit (azaz normalizált tokengyakoriságait) jegyezze fel, hanem a normalizált típusgyakoriságait. Ehhez kommentekben jelöltem hogy melyik sorokat kell átírni:

- a 90. és 91. sorokat (ahol a normalizáló konstanst kapjuk meg, az eredeti modell esetében a bigramok tokengyakoriságainak az összegét), és
- a 97. sort (ahol az eredeti modell esetében az unigram tokengyakoriságát kapjuk meg – itt az új modellben a típusgyakoriságát kellene megkapnunk).

A megváltoztatott változatot hasznos új függvényként definiálni, pl. `itp_type_model()` néven, hogy könnyen össze lehessen hasonlítani a két modellt.

### A h3.py-ban lévő függvények használata
Először töltsük le és helyezzük egy mappába a `h3.py` és a `grimm_full.txt` fájlokat és navigáljunk ebbe a mappába a parancssorban (vagy bármilyen más programozó felületen).

Miután elindítottuk a Python interpretert, importáljuk a `h3` modult:
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
Hozzuk létre a modellünket a tanítóadat alapján:
```
model = h3.itp_model(training_data)
```
Mérjük meg a modell perplexitását a tesztadaton, általunk választott súlyozással:
```
h3.perplexity(test_data, model, 0.75, 0.25)
```
(Itt a bigramvalószínűségeket súlyozzuk 0.75-tel és az unigramvalószínűségeket 0.25-tel.)
