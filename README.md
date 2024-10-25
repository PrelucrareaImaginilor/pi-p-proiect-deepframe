

![Diagramă bloc_v2](https://github.com/user-attachments/assets/90cabb96-57f7-4d50-8aa7-9fea2aefa8bc)
1.	Setul de date
   
	•	VisDrone https://github.com/VisDrone/VisDrone-Dataset

	•	Include imaginile foto-video pe baza cărora vom testa soluția noastră pe tot parcursul dezvoltării.

3.	Preprocesare
	•	Îmbunătățirea imaginilor pentru a asigura acuratețea rezultatelor și ușurința recunoașterii obiectelor;
	•	Aceasta presupune, în funcție de calitatea datelor, și nevoi îmbunătățirea contrastului, aplicarea de filtre pentru a elimina/limita blurarea imaginilor cauzată de mișcare sau evidențierea detaliilor.

4.	Detectarea obiectelor
	
	•	Vom folosi OpenCV, TensorFlow, YOLO pentru a detecta prezența obiectelor de interes (ex. autovehicule, clădiri, obstacole).

5.	Recunoașterea obiectelor

	•	Se folosesc resursele de mai sus, însă cu scopul de a diferenția și clasifica obiectele extrase din cadre (de exemplu folosind funcții din OpenCV).

6.	Urmărire (tracking)
	
	•	Avem la dispoziție mai multe opțiuni. De exemplu, Open CV oferă algoritmi cum ar fi KLT sau Optical Flow.

