# DreamBooth
>It’s like a photo booth, but once the subject is captured, it can be synthesized wherever your dreams take you…

Оригинальная статья: [DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/pdf/2208.12242.pdf)
## Проблематика и постановка задачи
Современные большие text-to-image модели могут достаточно точно и разнообразно генерировать изображения по текстовому запросу. Однако, возникает проблема, если мы хотим получить изображение с конкретным объектом, примеры которого у нас есть (3-5 изображений), но в измененном контексте, задаваемом промтом. Например, сгенерировать фотографию с собой/своим животным/любимой вещью в известном туристическом месте. Данную задачу позволяет решить подход DreamBooth.
## Основная идея
Тривиальный способ научить диффузионную модель генерировать заданный объект — это присвоить ему уникальный идентификатор (предлагается брать наиболее редко используемые токены, ведь от их оригинальной смысловой нагрузки будет проще всего избавиться, в промте будем обозначать как 'a \[V\] \[class noun\]'), чтобы модель понимала, что в промте речь идет про конкретный объект, а затем ее зафайнтьюнить как обычную диффузию. Функция потерь запишется как
```math
\mathbb{E}_{x, c, \epsilon, t} [w_t \|\hat{x}_\theta(\alpha_t x + \sigma_t \epsilon, c) - x\|_2^2],
```
где $x$ — истинное изображение, $c$ — вектор, полученный из текстового промта, $\epsilon \sim N(0, 1)$ — шум из нормального распределения, $\alpha_t, \sigma_t, w_t$ — величины, контролирующие шум и качество входного изображения, являются функциями времени $t \sim U([0, 1])$, $\hat{x}_\theta$ — диффузионная модель, предобученная удалять шум из зашумленного изображения $z_t := \alpha_t x + \sigma_t \epsilon$.
Такое решение действительно позволит создавать изображения с нашим объектом, однако, появляются две проблемы:
1. Language drift. Изначально появился в языковых моделях, предобученных на больших корпусах текста. После файнтьюна под узкую задачу такие модели переставали понимать синтаксис и семантику языка. При переносе на диффузионные модели, получаем, что нейросеть забывает, как должны строиться изображения, располагаться объекты по отношению друг к другу. Более того, весь класс, которому принадлежит наш объект, может у модели теперь ассоциироваться с конкретным его экземпляром.
2. Reduced output diversity. Text-to-image диффузионные модели выдают широкий спектр разнообразных изображений. Но после дообучения на небольшом наборе данных это разнообразие теряется, и мы больше не в состоянии получить фотографию желаемого объекта в другой позе или с другого ракурса.

Побороть обе проблемы можно, если дообучать не только на небольшом датасете с интересуемым объектом, но и на изображениях, генерируемых моделью до файнтьюна по промту, содержащему лишь класс объекта. Таким образом, мы и сохраняем разнообразие и не переобучаемся на маленькой выборке.

Лосс-функция теперь перепишется как
```math
\mathbb{E}_{x, c, \epsilon, \epsilon', t} [w_t \|\hat{x}_\theta(\alpha_t x + \sigma_t \epsilon, c) - x\|_2^2 + \lambda w_{t'} \|\hat{x}_\theta(\alpha_{t'} x_{pr} + \sigma_{t'} \epsilon', c_{pr}) - x_{pr}\|_2^2],
```
здесь все обозначения совпадают с предыдущей формулой, за исключением: 
* $x_{pr} = \hat{x}(z_{t_1}, c_{pr})$ — данные, сгенерированные предобученной диффузионной моделью с замороженными весами из шума $z_{t_1} \sim N(0, 1)$
* $c_{pr} := \Gamma(f('\text{a [class noun]}'))$ — вектор на выходе энкодера текста
* $\lambda$ — параметр, контролирующий отношение слагаемых.

Первое слагаемое назовем Reconstruction Loss, второе — Class-Specific Prior Preservation Loss. Измененный процесс файнтьюна можно изобразить в виде схемы:
![Fine-Tune scheme](images/fine-tune_scheme.png)
## Результаты из оригинальной статьи
![Prior-Preservation Loss comparison](images/prior-preservation_loss.png)
Как можно заметить, на изображениях входного датасета собака лежит на мягких поверхностях, и на сгенерированных изображениях без prior-preservation loss'а она тоже лежит на похожих поверхностях. А на изображениях, полученных с prior-preservation loss'ом, собака стоит и сидит на отличающихся поверхностях.

Также авторам удалось достичь успеха в следующих задачах синтеза изображений:
* Recontextualization — изменение окружения объекта
  ![Recontextualization](images/recontextualization.png)
* Novel View Synthesis — синтез новых ракурсов
  ![Novel View Synthesis](images/novel_view_synthesis.png)
* Art Renditions — генерация в стиле картин великих художников
  ![Art Renditions](images/art_renditions.png)
* Property Modification — изменение качеств объекта
  ![Property Modification](images/property_modification.png)


## Дообучение LoRA под генерацию своих объектов
В качестве домена я выбрал фотографии кота своего друга. Вот несколько из них:
![Cat](/train_data/instance/1.png)
![Cat](/train_data/instance/4.png)
Параметры дообучения:
* Количество шагов - 500
* Ранг LoRA - 16 (больше смысла не имеет, меньше может быть недостаточно, чтобы сдвинуть домен)
По итогу получилась модель, способная генерировать наш объект в различных контекстах.
Теперь посмотрим, как меняется результат на инференсе в зависимости от гиперпараметров. Их можно выделить 3: LORA_SCALE_UNET, LORA_SCALE_TEXT_ENCODER, GUIDANCE. Первые два отвечают за то, на какой коэффициент будет умножаться прибавка от LoRA, то есть параметру 0 соответствует предобученная модель, а параметру 1 — модель, полученная после файнтьюна. GUIDANCE — насколько близко к промту будет сгенерированное изображение. Чем больше GUIDANCE, тем более уникальные и разнообразные изображения будут получаться, однако, качество самих изображений будет ухудшаться.
Посмотрим, как влияют параметры LORA_SCALE*, значение GUIDANCE будет зафиксировано примерно посередине разумного диапазона числом 7.5.
![LoRAScale](images/inference/LoRAScale.png)
Здесь сверху вниз растет LORA_SCALE_UNET, слева направо LORA_SCALE_TEXT_ENCODER. Как можно заметить, при генерации изображения по стандартному промту "A photo of a [V] cat", значения коэффициента энкодера не влияют. Но выберем оптимальные по качеству значения коэффициентов (0.8, 0.8).
Теперь посмотрим на влияние GUIDANCE при уже заданных LORA_SCALE:
![LoRAGuidance](mages/inference/LoRAGuidance.png)
При значениях больше 9 становится хуже качество изображений, появляются артефакты. А при значениях меньше 6 изображения достаточно похожи на таковые из референса. Выберем оптимальным значение GUIDANCE равное 7.
Теперь зная хорошие значения на инференсе, попробуем погенерировать изображения с разными промтами.

### Проверка на переобучение всего домена "cat"
Чтобы убедиться, что наша модель не забыла, как выглядят коты в общем случае, запустим генерацию изображений несколько раз на промте "A photo of a cat". Вот полученные картинки:
![Cat1](images/inference/Pic-A_photo_of_a_cat-1.jpg)
![Cat2](images/inference/Pic-A_photo_of_a_cat-2.jpg)
![Cat3](images/inference/Pic-A_photo_of_a_cat-3.jpg)
Полученные коты отличаются от нашего заданного цветом шерсти и рисунком на ней.

### Изменение контекста
Теперь посмотрим, как модель генерирует нашего кота в различных контекстах.
#### Recontextualization
Попробуем генерацию изображений по промтам вида "A photo of a [V] cat in a [place]".
* "A photo of a [V] cat in a bath"
  
  ![Bath](images/inference/Pic-A_photo_of_a_[V]_cat_in_a_bath.jpg)
* "A photo of a [V] cat driving a car"
  
  ![Driving](images/inference/Pic-A_photo_of_a_[V]_cat_driving_a_car.jpg)
* "Pic-A photo of a [V] cat on a moon surface.jpg"
  
  ![Moon surface](Pic-A_photo_of_a_[V]_cat_on_a_moon_surface.jpg)
* "A photo of a [V] cat in a snow."
  
  ![Snow](images/inference/Pic-A_photo_of_a_[V]_cat_in_a_snow.jpg)
#### Property Modification
Попробуем повторить результаты статьи, где они скрещивали таргетный объект с другими животными. Вот несколько полученных изображений.
* "A photo of a [V] cat crossed with a hippo"
  
  ![A photo of a V cat crossed with a hippo](images/inference/Pic-A_photo_of_a_[V]_cat_crossed_with_a_hippo.jpg)
* "A photo of a [V] cat crossed with a panda"
  
  ![Panda](images/inference/Pic-A_photo_of_a_[V]_cat_crossed_with_a_panda-1.jpg)
* "A photo of a [V] cat crossed with a koala"
  
  ![Koala](images/inference/Pic-A_photo_of_a_[V]_cat_crossed_with_a_koala.jpg)
* "A photo of a [V] cat crossed with a lion"
  
  ![Koala](images/inference/Pic-A_photo_of_a_[V]_cat_crossed_with_a_lion.jpg)
#### Expression modification
* "A photo of a depressed [V] cat"
  
  ![depressed](images/inference/Pic-A_photo_of_a_depressed_[V]_cat.jpg)
* "A photo of a sad [V] cat"
  
  ![sad](images/inference/Pic-A_photo_of_a_sad_[V]_cat.jpg)
* "A photo of a happy [V] cat"
  
  ![depressed](images/inference/Pic-A_photo_of_a_happy_[V]_cat.jpg)
* "A photo of a screaming [V] cat"
  
  ![screaming](images/inference/Pic-A_photo_of_a_screaming_[V]_cat.jpg)
#### Novel View Synthesis 
Попробуем получить другие ракурсы нашего кота.
* "A photo of a [V] cat seen from the back"
  
  ![back](images/inference/Pic-A_photo_of_a_[V]_cat_seen_from_the_back.jpg)
* "A photo of a [V] cat seen from the bottom"
  
  ![bottom](images/inference/Pic-A_photo_of_a_[V]_cat_seen_from_the_bottom.jpg)
* "A photo of a [V] cat seen from the side"
  
  ![side](images/inference/Pic-A_photo_of_a_[V]_cat_seen_from_the_side.jpg)
* "A photo of a [V] cat seen from the top"
  
  ![top](images/inference/Pic-A_photo_of_a_[V]_cat_seen_from_the_top.jpg)

### Выводы
Полностью повторить результаты оригинальной статьи не получилось. Объяснить это можно тем, что мы обучали LoRA, а не дообучали всю модель. К тому же можно было генерировать больше классовых изображений и возможно от этого получить большее разнообразие. Однако, удалось связать "[V]" в промте с конкретными параметрами объекта и получить изображения, сильно отличающиеся от данных в reference set. При этом не был смещен весь домен класса. В поисках лучшего результата можно поэкспериментировать с количеством шагов на обучении и на инференсе. Код можно найти в ноутбуке. Также можно поварьировать коэффициент, меняющий соотношение между Reconstruction Loss и Class-Specific Prior Preservation Loss.