# A-14-Hometask-arasdirma
PyTorch və Neyron Şəbəkələrinin Əsasları
1. PyTorch və neyron şəbəkələri
PyTorch Python üçün hazırlanmış açıq mənbəli dərin öyrənmə kitabxanasıdır. Tensor əməliyyatlarını, avtomatik gradient hesablamasını və GPU ilə sürətli hesablamanı dəstəkləyir. Buna görə PyTorch neyron şəbəkələrinin qurulması, öyrədilməsi və test edilməsi üçün geniş istifadə olunur.

Neyron şəbəkəsi giriş məlumatlarını bir və ya bir neçə gizli laydan keçirərək çıxış yaradan riyazi modeldir. Hər neyron girişləri çəkilərlə vurur, bias əlavə edir və aktivasiya funksiyasından keçir. Öyrənmə zamanı çəkilər elə dəyişdirilir ki, modelin səhvi azalsın.

torch.nn PyTorch-da neyron şəbəkəsi komponentlərini saxlayan moduldur. nn.Module bütün modellər üçün əsas sinifdir. nn.Linear tam əlaqəli laydır və y = xW^T + b hesablamasını aparır.

forward() metodu giriş tensorunun modelin laylarından necə keçəcəyini müəyyən edir. Model çağırıldıqda, məsələn model(x), PyTorch avtomatik olaraq forward(x) metodunu işə salır.

Nümunə:
import torch
from torch import nn

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
        )

    def forward(self, x):
        return self.layers(x)
2. Loss funksiyaları
Loss function modelin proqnozu ilə düzgün cavab arasındakı fərqi ölçür. Loss kiçildikcə modelin təlim məlumatına uyğunluğu artır. Optimizer bu loss-un gradientlərindən istifadə edərək model parametrlərini yeniləyir.
MSELoss
Mean Squared Error əsasən regression məsələlərində istifadə olunur:

MSE = (1/n) * sum((y_pred - y_true)^2)

Böyük səhvləri kvadrata yüksəltdiyi üçün outlier-lərə həssasdır.
loss_fn = nn.MSELoss()
BCELoss
Binary Cross Entropy iki sinifli təsnifat üçün istifadə olunur. Model çıxışında hər nümunə üçün 0 ilə 1 arasında ehtimal olmalıdır. Bu səbəbdən çox vaxt son layda Sigmoid istifadə edilir.
model = nn.Sequential(nn.Linear(4, 1), nn.Sigmoid())
loss_fn = nn.BCELoss()
Praktikada Sigmoid + BCELoss əvəzinə BCEWithLogitsLoss istifadə etmək daha stabil ola bilər. Bu halda modelin son layında Sigmoid yazılmır.
CrossEntropyLoss
Cross Entropy çoxsinifli təsnifat üçün istifadə olunur. Model hər sinif üçün xam score, yəni logits qaytarır. CrossEntropyLoss daxilində LogSoftmax və NLLLoss əməliyyatlarını birlikdə həyata keçirir.
model = nn.Linear(4, 3)
loss_fn = nn.CrossEntropyLoss()
Modelin çıxışı [batch_size, number_of_classes] ölçüsündə olur. Target isə sinif indekslərindən ibarət LongTensor olmalıdır.
3. Softmax və Cross Entropy
Softmax xam logits dəyərlərini ehtimallara çevirir:

softmax(z_i) = exp(z_i) / sum(exp(z_j))

Nəticədə bütün ehtimallar 0 ilə 1 arasında olur və cəmi 1-ə bərabərdir. Ən böyük ehtimala sahib sinif modelin seçdiyi sinifdir.
logits = torch.tensor([[2.0, 1.0, 0.1]])
probabilities = torch.softmax(logits, dim=1)
prediction = probabilities.argmax(dim=1)
Cross Entropy düzgün sinfin ehtimalı yüksək olduqda kiçik, aşağı olduqda isə böyük loss verir:

CE = -log(p_correct)

Softmax ilə Cross Entropy əlaqəsi:
1. Model logits yaradır.
2. Softmax logits-i ehtimallara çevirir.
3. Cross Entropy düzgün sinfin ehtimalını qiymətləndirir.
4. Gradient həmin səhvə əsasən çəkiləri yeniləyir.
PyTorch-da CrossEntropyLoss Softmax-ı daxilində hesabladığı üçün modelə əvvəlcədən Softmax əlavə etmək lazım deyil.
logits = model(x)
loss = nn.CrossEntropyLoss()(logits, labels)
4. Optimizer-lər
Optimizer modelin çəkilərini gradientlər əsasında yeniləyən alqoritmdir. Gradient Descent-in əsas formulu:

w_new = w_old - learning_rate * gradient

Learning rate hər addımda parametrlərin nə qədər dəyişəcəyini göstərir. Çox böyük learning rate öyrənməni qeyri-sabit edə bilər, çox kiçik learning rate isə təlimi yavaşlada bilər.
SGD
SGD sadə və klassik optimizer-dir:
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
Adam
Adam hər parametr üçün gradientin orta qiymətini və kvadrat gradientin orta qiymətini izləyir:
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
Təlim dövrü
loss.backward() loss-un parametrlərə görə gradientlərini hesablayır. optimizer.step() həmin gradientlərdən istifadə edib çəkiləri yeniləyir. optimizer.zero_grad() əvvəlki addımın gradientlərini silir.
for x_batch, y_batch in train_loader:
    optimizer.zero_grad()
    logits = model(x_batch)
    loss = loss_fn(logits, y_batch)
    loss.backward()
    optimizer.step()
Əsas fərqlərin müqayisəsi
Mövzu	Əsas məqsəd	Tipik istifadə
MSELoss	Ədədi səhvi ölçmək	Regression
BCELoss	İki sinif ehtimalını ölçmək	Binary classification
CrossEntropyLoss	Çoxsinifli logits-i qiymətləndirmək	Multiclass classification
Softmax	Logits-i ehtimala çevirmək	Proqnozu izah etmək
SGD	Gradient ilə sadə yeniləmə	Nəzarətli optimizasiya
Adam	Adaptiv gradient yeniləməsi	Ümumi deep learning


Dərs üçün hazır cavablar
PyTorch nədir? Tensor hesablaması, avtomatik differentiation və neyron şəbəkəsi qurmaq üçün Python kitabxanasıdır.

nn.Module nə üçün istifadə olunur? Modelin laylarını və öyrənilə bilən parametrlərini strukturlaşdırmaq üçün əsas sinifdir.

forward() nə edir? Girişin modelin laylarından keçərək çıxışa çevrilmə qaydasını müəyyən edir.

Loss niyə lazımdır? Modelin səhvini rəqəmlə ölçür və optimizer üçün öyrənmə siqnalı yaradır.

BCELoss ilə CrossEntropyLoss fərqi nədir? BCELoss binary classification və ehtimal çıxışı üçün, CrossEntropyLoss isə çoxsinifli classification və logits üçün istifadə olunur.

Softmax nə edir? Logits dəyərlərini cəmi 1 olan ehtimallara çevirir.

CrossEntropyLoss istifadə edərkən Softmax yazmaq lazımdırmı? Xeyr. CrossEntropyLoss Softmax və log əməliyyatını özü hesablayır.

loss.backward() nə edir? Loss-un model parametrlərinə görə gradientlərini hesablayır.

optimizer.step() nə edir? Hesablanmış gradientlərdən istifadə edərək model çəkilərini yeniləyir.

Learning rate nədir? Parametrlərin bir gradient addımında nə qədər dəyişəcəyini təyin edən hiperparametrdir.
Nəticə
Neyron şəbəkəsinin quruluşu nn.Module, nn.Linear və forward() vasitəsilə ifadə olunur. Loss funksiyası məsələnin növünə uyğun seçilməlidir: regression üçün MSELoss, binary classification üçün BCELoss və çoxsinifli classification üçün CrossEntropyLoss uyğundur. Softmax logits-i ehtimallara çevirsə də, CrossEntropyLoss istifadə edərkən onu modelin sonuna ayrıca əlavə etmək lazım deyil. Optimizer gradientlərdən istifadə edərək parametrləri yeniləyir. SGD daha sadə, Adam isə adaptiv və çox vaxt daha sürətli optimizer-dir. Düzgün learning rate seçimi modelin sabit və sürətli öyrənməsi üçün vacibdir.
