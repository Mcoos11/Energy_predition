# Energy-predictor
1. Dane z daily_dataset umieścić w data
2. Uruchomić maked_dataset.py - tworzenie zbiorów danych dla modeli, osobny dla każdego gospodarstwa
3. Uruchomić make_models.py - tworzenie modeli Random Forest i K Near Neighbors, tworzenie zbiorów testowych na podstawie danych uczących dla każdego gospodarstwa
4. Uruchomić prediction.py - odczyt zapisanych modeli i danych testowych, predykcja, oblicznie błędu średniokwadratowego predykji

<b>UWAGA</b>
Danych jest dość sporo. Na razie ustawione jest (make_datasets.py:35) aby predykcja była tylko dla 50 gospodarstw tj. tylko z pliku block_0.csv. Przy takiej liczbie gospodarstw to i tak chwilę trwa ale udało mi się przyspieszyć ten proces dodając multiprocessing. Warto też wrzucić to na jakiś szybki dysk SSD to dodatkowo przyspiesza bo jest dużo plików do odczytu i zapisu. U mnie całość z danymi, modelmi i wynikami waży 15,6GB.
