# Classibundler

## Brain Bundle Analisys for Humans

Is an extention of Dipy package, including

- A better bundle classifier  
- An API for managing patients  
- Tools for data visualization


## Checklist

- [x] Выключать пациентов из группы, если в имени есть `!`  
- [x] Строить графики для FA, AD, RD, MD  
- [x] График p-value  
- [ ] Референсные линии для p-value  
- [ ] Вынести p-value на отдельный график  
- [x] Рисовать пучки по каждому пациенту  
- [ ] Волокна пучков должны быть выровнены  
- [x] Подрезать концы пучков  
- [x] Конфиг фич, по которым проводить измерения  
- [x] Участок, на котором видим отклонения, выделить цветом для статьи/диссера  
- [ ] Записывать измерения в табличку  
- [ ] Нарисовать пучки атласа  
- [ ] Right/Left патологии  
- [ ] Еще раз пройтись по атласу  
- [ ] Добиться, чтобы ориентация волокон выбиралась в Classibundler  
- [ ] Сравнить с количественными симптомами пациентов  
- [ ] Предобработка DICOM-ов  
- [ ] Мультипроцессинг на маке  
- [?] Усреднять с весами Гаусса  
- [ ] Рисовать среднее нормы на каждом пациенте  


## Notes

## Задача

#### Волокна
По МРТ в каждой точке головы измеряется тензор диффузии, из которого можно вытащить основное направление. Это дает нам трехмерное поле направлений. Если решать диффур и шагать вдоль этих направлений, получается линия тока. У пациента в мозге мы находим линии тока и считаем, что это нервные волокна. В нашей модели волокно - отрезок бесконечно-тонкой кривой, немного извитой. В мозгу конкретного пациента мы находим 100-200 тысяч таких волокон, но это условное число, потому что их можно найти сколько угодно, просто 100к вполне достаточно, и столько же считают достаточным наши западные колеги.

#### Пучки
Мы также считаем, что волокна организованы в пучки (тракты) - это набор похожих близко расположенных волокон. У каждого пучка есть анатомическое название - их несколько сотен, но интересных (и легко находимых) из них пара десятков. Пример https://radiopaedia.org/articles/forceps-major

#### Атлас
У нас есть модельный пациент (атлас, одна из нескольких общепринятых в сообществе моделей), на котором эти пучки размечены - для каждого волокна отмечено название пучка, к которому он принадлежит. Тут надо заметить, что это тоже всё весьма условно: определение пучка - это соединение между двумя зонами мозга, но зоны мозга могут определены весьма размыто. Из атласа мы достаем ориентацию пучка - все волокнав пучке должны быть сориентированы в одном направлении.

В атласе есть достаточно багов - ошибки, размечены не все волокна, которые должны быть в пучке и тп. Также в атласе есть неразмеченные волокна, которые либо из каких-то мелких неинтересных пучков, либо вовсе не являются нервными волокнами - всякие сосуды, шум и прочий мусор.

#### Классификатор
Итак, у нас есть волокна, найденные у пациента, и волокна, размеченные в атласе. Нехитрым сопоставлением программка автоматически классифицирует каждое волокно в мозге пациента как принадлежащее к тому или иному пучку. Тут есть нюансы - волокна могут классифицироваться с ошибками, а некоторые пучки вообще не найдутся. У меня есть оценка точности этого классификатора: порядка 95% intersection over union, но это грубая оценка, потому что разметка сама наполовину состоит из багов, и каждый пациент уникален, пучки могут довольно сильно гулять - даже люди не сильно заморачиваются с тем, чтобы идеально точно их отметить.

#### Профиль
Мы идем по МРТ-скану вдоль каждого волокна и выбираем из этого скана значения. x - криволинейная координата вдоль волокна, можно измерять в миллиметрах или в % от длины волокна, y - интенсивность изображения в точке y = y(R(x)), R - уравнение кривой нашего волокна, задано просто как последовательность координат. Так получается профиль волокна y(x). Если то же самое проделать для каждого волокна в пучке, получится профиль пучка - его среднее и std. Нюанс: каждое волокно в пучке отличается - они проходят чуть в разных местах и имеют немного разную длину. У разных пациентов в одном и том же пучке может быть разное количество волокон, они могут быть немного разной длины и конфигурации.

#### Трактограмма
Если построить профиль для каждого пучка у пациента, получается трактограмма - пачка графиков с циферками.

#### Группы пациентов
Теперь мы хотим понять, можно ли по трактограмме отличить больного и здорового. Очевидно, пациентов поделили на здоровых и больных, и теперь трактограммы этих групп надо както сравнить


## Графики
ТР: Очень хорошо, что программа Вам строит графики, я всегда пропагандирую визуализировать данные перед любым анализом. Возникают следующие мысли. Что разные измерения по оси абсцисс никак нельзя назвать случайными. Т.е. мы наблюдаем некие патерны изменения параметров при движении от 0 до 100. Соответственно подвопрос 1 - с каким шагом измерялись индивидуальные значения, подвопрос 2 - нужно ли их учитывать при расчете среднего и дисперсии как вложенный фактор, подвопрос 3 - устойчивы ли результаты к изменению параметров измерения (количество точек, шаг и т.п.), подвопрос 4 - не могут ли быть какие-либо участки пучков по измеряемым показателям более информативны, чем другие. И т.д.

1. Шаг выбирается так, чтобы координата измерялась в процентах - от 0 до 100. шаги расположены равномерно по длине волокна, и примерно на одном срезе пучка (см картику по этой ссылке)  
2. Думаю, что нет, но я, возможно, не понял вопрос  
3. Устойчивы, но некоторая дополнительная стратификация волокон, наверное, не помешает  
4. Скорее всего концы пучка менее информативны, и некоторые авторы действительно только серединку и учитывают. Возможно, следует еще и только на серцевину пучка смотреть (волокна внутри пучка, которые ближе к середине сечения), но тут не ясно, как с водой ребеночка не выплеснуть


## Статистика

1. Мы можем сами по желанию левой пятки нарезать эти графики на куски, которые нам нравятся, посчитать по ним какие-нибудь штуки типа min, max, средне-геометрическое, регрессию и тп, назвать их сущностями и анализировать  
2. Найти такие подходящие сущности нам придется глазками  
3. Если есть какие-то вопросы к параметрам - например к параметрам классификатора, обрезанию концов и тп, эти параметры нужно просто изолировать и посмотреть отдельно


## List of bundles

AC 	клешня - х  
AF_L  - доп  
AF_R  - доп  
AR_L - х  
AR_R  х  
AST_L - доп  
AST_R - доп  
CB_L - мозжечок доп  
CB_R - мозжечок доп  
CC - х  
CC_ForcepsMajor - да  
CC_ForcepsMinor  -да  
CC_Mid - да  
CNIII_L  - х  
CNIII_R - х  
CNII_L - х  
CNII_R - х  
CNIV_L - х  
CNIV_R - х  
CNVIII_L - х  
CNVIII_R - х  
CNVII_L - х  
CNVII_R - х  
CNV_L - х  
CNV_R - х  
CST_L  - да  
CST_R   - да  
CS_L  - х  
CS_R  - х  
CTT_L  - х  
CTT_R  - х  
CT_L  - х  
CT_R  - х  
C_L  - х  
C_R  - х  
DLF_L - х  
DLF_R  - х  
EMC_L  - доп или еще понабл  
EMC_R  - доп или еще понабл  
FPT_L - да или допчик  
FPT_R - да или допчик  
F_L_R  - х  
ICP_L - х  
ICP_R - х  
IFOF_L - да  
IFOF_R - да  
ILF_L  - да Маше х - Лехе  
ILF_R  - да Маше х - Лехе  
LL_L  -х  
LL_R  -х  
MCP - набл  
MLF_L - х  
MLF_R - х  
ML_L - х  
ML_R - х  
MdLF_L - доп  
MdLF_R - доп  
OPT_L - х  
OPT_R - х  
OR_L - х  
OR_R - х  
PC  - х  
PPT_L - доп  
PPT_R - доп  
RST_L - х  
RST_R - х  
SCP  - х  
SLF_L - х  
SLF_R - х  
STT_L - х  
STT_R - х  
TPT_L - х  
TPT_R - х  
UF_L - да  
UF_R - да  
V  - х  
VOF_L  - х  
VOF_R  - х  
