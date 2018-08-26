Deep learning (3): PyTorch và MNIST
-----------------------------------

Tác giả: Nguyễn Xuân Khánh

**Nguồn**: [Khánh's blog] (http://khanhxnguyen.com/deep-learning-3/)


Chào các bạn, hôm nay chúng ta sẽ nghiên cứu cấu trúc một chương trình deep learning cơ bản. Nếu khi thực hành có khó khăn, các bạn comment vào dưới bài viết hoặc vào [group machine learners](https://www.facebook.com/groups/485581088316769/) để được hỗ trợ.

0\. PyTorch:

Đầu tiên, chúng ta cần chọn một nền tảng deep learning để thực hành. Mình chọn [PyTorch](https://github.com/pytorch/pytorch) vì nó có những ưu điểm sau:

-- Thân thiện: PyTorch cho phép theo dõi được các tham số của model trong khi chạy, giúp debug được thuận tiện hơn.

-- Linh động: PyTorch cho phép thay đổi cấu trúc model khi chạy.

-- Large scale: PyTorch hỗ trợ chạy với nhiều GPU một cách đơn giản.

-- Đội ngũ phát triển: PyTorch được Facebook, Twitter và các trường đại học lớn phát triển vì thế sẽ luôn cập nhật những công nghệ đỉnh nhất từ giới nghiên cứu.

1\. Cài đặt PyTorch:

Mình mặc định là Ubuntu và Python đã được cài sẵn.

a. Cài đặt CUDA và CuDNN:

Lựa chọn này chỉ dành cho người có GPU.

[Ở đây](https://yangcha.github.io/GTX-1080/) có một hướng dẫn chi tiết để cài đặt máy với GPU GTX 1080. Các bạn nhớ thay đổi tên file tùy theo version của CUDA và CuDNN được tải về. Nếu các bạn đã hoàn thành hết các bước trong đó thì chuyển qua mục (b) luôn mà không cần đọc tiếp.

Dưới đây mình chỉ cách tải các file CUDA và CuDNN về sao cho đúng.

Để cài đặt CUDA, các bạn google CUDA rồi vào trang chủ, chọn cấu hình máy cho phù hợp rồi tải file về. Mình thì thường hay tải file .deb về vì nó dễ cài đặt. Lưu ý là file khá nặng (gần 2GB).

![](https://i2.wp.com/khanhxnguyen.com/wp-content/uploads/2017/03/Selection_013.png?resize=640%2C747)

Sau khi tải xong, các bạn vào thư mục có file vừa tải và chạy 3 dòng lệnh như hướng dẫn trong terminal.

|

1

2

3

 |

`sudo` `dpkg -i FILE_CUDA.deb`

`sudo` `apt-get update`

`sudo` `apt-get` `install` `cuda`

 |

Lưu ý là dòng lệnh đầu thay đổi tùy theo tên file nên tốt nhất cứ copy paste vào terminal nhé.

CuDNN đòi hỏi các bạn phải đăng ký, điền thông tin và đợi email chấp nhận mới tải được. Trang tải CuDNN sẽ trông như sau:

![](https://i1.wp.com/khanhxnguyen.com/wp-content/uploads/2017/03/Selection_015.png?resize=552%2C280)

Mình tải về gói được gạch dưới trong hình trên. Sau khi tải về, các bạn chạy những câu lệnh sau:

|

1

2

3

4

 |

`tar` `xvzf FILE_CUDNN.tgz`

`sudo` `cp` `cuda``/include/cudnn``.h` `/usr/local/cuda/include`

`sudo` `cp` `cuda``/lib64/libcudnn``*` `/usr/local/cuda/lib64`

`sudo` `chmod` `a+r` `/usr/local/cuda/include/cudnn``.h` `/usr/local/cuda/lib64/libcudnn``*`

 |

b. Cài đặt PyTorch:

Các bạn vào [trang chủ của PyTorch](http://pytorch.org/) và lựa chọn cách cài đặt cho phù hợp. Ví dụ mình hay lựa chọn cách cài như sau:

![](https://i0.wp.com/khanhxnguyen.com/wp-content/uploads/2017/03/Selection_017.png?resize=640%2C347)

Sau đó các bạn copy 2 dòng lệnh ở ô "run this command" và chạy chúng trong terminal. Nếu khi cài các bạn bị báo lỗi chưa cài package manager nào thì google cách cài package manager đó nhé (ví dụ google "install pip ubuntu").

Còn một cách cài nữa đó là các bạn clone repo của PyTorch về máy (clone là một thuật ngữ của [git](http://rogerdudler.github.io/git-guide/), chỉ việc download code về máy) bằng cách mở terminal ra và gõ dòng lệnh sau:

|

1

 |

`git clone https:``//github``.com``/pytorch/pytorch``.git`

 |

Sau đó làm theo hướng dẫn [ở đây](https://github.com/pytorch/pytorch#installation). Hướng dẫn này bao gồm cả việc cài đặt không cần CUDA. Cách bạn chỉ cần định nghĩa một biến NO_CUDA trong terminal trước khi cài đặt.

|

1

 |

`export` `NO_CUDA=1`

 |

2\. Tải code ví dụ:

Để học PyTorch hiệu quả, các bạn nên bỏ thời gian ra xem các [code ví dụ mẫu](https://github.com/pytorch/examples) được các chuyên gia PyTorch viết sẵn.

Ở đây, mình copy ví dụ MNIST về và sửa lại để đơn giản hóa một số vấn đề tải dữ liệu. Các bạn clone repo [mnist-pytorch](https://github.com/khanhptnk/mnist-pytorch) của mình về máy:

|

1

 |

`git clone https:``//github``.com``/khanhptnk/mnist-pytorch``.git`

 |

Sau đó trong thư mục hiện hành của bạn sẽ xuất hiện thư mục "mnist-pytorch". Kiểm tra điều đó bằng cách dùng lệnh:

|

1

 |

`ls` `. |` `grep` `mnist-pytorch`

 |

Sau đó các bạn đi vào thư mục này:

|

1

2

 |

`cd` `mnist-pytorch`

`ls` `-1`

 |

3\. MNIST:

MNIST là một bài toán nhận diện chữ số viết tay thông qua hình ảnh. Input vào là một ảnh trắng đen của một chữ số viết tay từ 0 đến 9. Nhiệm vụ của model là dự đoán xem tấm ảnh đó biểu thị số nào. Ví dụ như đây là số 2:

![](https://i2.wp.com/khanhxnguyen.com/wp-content/uploads/2017/03/mnist-2.png?resize=165%2C166)

MNIST được sử dụng như một bài tập dạng "Hello world" cho deep learning. MNIST là một bài toán kinh điển về multiclass classification, tức là phân loại có nhiều loại nhãn. Multiclass classification là một dạng supervised learning, tức là mỗi input xxđược gắn với một loại label (nhãn) yy. Label yy được cho biết trong lúc huấn luyện. Label yy nhận các giá trị từ tập rời rạc YY có nhiều hơn 2 phần tử  (nếu YY chỉ có hai phần tử người ta gọi là binary classification). Trong trường hợp ở đây thì xx chính là tấm ảnh, yy là con số tấm ảnh biểu thị, còn YY là tập hợp các số từ 0 đến 9.

Trong thư mục "mnist-pytorch", bạn chạy thử chương trình như sau:

|

1

 |

`python main.py`

 |

Nếu chạy thành công bạn sẽ nhìn thấy output như sau:

![](https://i2.wp.com/khanhxnguyen.com/wp-content/uploads/2017/03/Selection_016.png?resize=640%2C798)

Giải thích output này một chút. Model sẽ đi qua hết các ví dụ của tập train nhiều lần. Mỗi lần như vậy được gọi là một epoch. Trong mỗi epoch, để tiết kiệm bộ nhớ tập train sẽ được chia nhỏ thành các batch. ở đây các bạn thấy là tập train gồm 60000 ví dụ. Batch size, tức là số lượng ví dụ trong mỗi batch, là 64. Vậy thì để đi hết các ví dụ ta cần 60000 / 64 batch. Batch thứ nhất gồm các ví dụ từ 1 đến 64, batch thứ hai từ 65 đến 128, và cứ thế. Batch cuối cùng thì có thể nhỏ hơn các batch còn lại nếu như số lượng ví dụ không chia hết cho batch size. Sở dĩ phải chia thành batch như vậy là vì bộ nhớ của máy tính có hạn. Các bạn dùng batch size=64 là đủ tốt cho đa số các ứng dụng. Cũng lưu ý là batch size thường sẽ là một lũy thừa nào đó của 2 để sử dụng bộ nhớ được hiệu quả.

Chương trình in ra loss, tức là giá trị của [hàm mất mát](http://khanhxnguyen.com/loss-function/), sau mỗi 200 batch (log interval bằng 200). Vì batch size là 64, nên nói cách khác, loss được sau mỗi 200 * 64 = 12800 ví dụ. Hàm mất mát được dùng ở đây là gì mình sẽ nói trong bài sau. Hiện giờ các bạn chỉ cần hiểu nó thể hiện cho độ chính xác của model khi huấn luyện. Lưu ý là loss ở đây chỉ là giá trị hàm mất mát cho batch hiện thời thôi, không phải của cả tập train. Sau cùng chúng ta mới in ra average loss trên train set, được tính bằng trung bình cộng của các loss của tất cả các batch.

Sau mỗi epoch, chúng ta in ra average loss trên tập test. Đây là trung bình giá trị hàm mất mát trên tất cả các ví dụ của tập test, chứ không phải tập train. Nếu các bạn thắc mắc tại sao lại là tập test, hãy xem lại quy trình supervised learning ở [bài này](http://khanhxnguyen.com/overfit-2/). Lưu ý là vì ở đây chúng ta không có hyperparameter nên không cần tập dev. Chúng ta đánh giá model trên tập test luôn. Quan sát average loss trên tập train và tập test giúp chúng ta xác định xem model có bị [overfit](http://khanhxnguyen.com/machine-learning-101-overfit/) hay không. Ở đây ta thây cả hai đều đang giảm sau mỗi epoch, tức là model đang không bị overfit.

Tuy nhiên, với bài toán này, giá trị chúng ta quan tâm hơn khi so sánh độ tốt giữa các model là accuracy. Accuracy là phần trăm số lượng ví dụ trong tập test mà model đoán đúng label. Vì MNIST là một bài toán khá dễ nên model đạt accuracy gần như tuyệt đối (98%) chỉ sau 5 epoch.

4\. Thay đổi flag:

Câu hỏi đặt ra là làm để thay các thông số như là số lượng epoch hay batch size? Nếu các bạn nhìn vào file [main.py](https://github.com/khanhptnk/mnist-pytorch/blob/master/main.py) trong thư mục "mnist-pytorch" vừa tải về thì các dòng đầu tiên sẽ trông giống như thế này:

|

1

2

3

4

5

6

7

8

9

10

11

12

13

14

15

16

17

18

19

20

21

 |

`parser` `=` `argparse.ArgumentParser(description``=``'PyTorch MNIST Example'``)`

`parser.add_argument(``'--no_download_data'``, action``=``'store_true'``, default``=``False``,`

`help``=``'Do not download data'``)`

`parser.add_argument(``'--batch-size'``,` `type``=``int``, default``=``64``, metavar``=``'N'``,`

`help``=``'input batch size for training (default: 64)'``)`

`parser.add_argument(``'--test-batch-size'``,` `type``=``int``, default``=``1000``, metavar``=``'N'``,`

`help``=``'input batch size for testing (default: 1000)'``)`

`parser.add_argument(``'--epochs'``,` `type``=``int``, default``=``5``, metavar``=``'N'``,`

`help``=``'number of epochs to train (default: 10)'``)`

`parser.add_argument(``'--lr'``,` `type``=``float``, default``=``0.01``, metavar``=``'LR'``,`

`help``=``'learning rate (default: 0.01)'``)`

`parser.add_argument(``'--momentum'``,` `type``=``float``, default``=``0.5``, metavar``=``'M'``,`

`help``=``'SGD momentum (default: 0.5)'``)`

`parser.add_argument(``'--no-cuda'``, action``=``'store_true'``, default``=``False``,`

`help``=``'enables CUDA training'``)`

`parser.add_argument(``'--seed'``,` `type``=``int``, default``=``1``, metavar``=``'S'``,`

`help``=``'random seed (default: 1)'``)`

`parser.add_argument(``'--log-interval'``,` `type``=``int``, default``=``200``, metavar``=``'N'``,`

`help``=``'how many batches to wait before logging training status'``)`

`args` `=` `parser.parse_args()`

`args.cuda` `=` `not` `args.no_cuda` `and` `torch.cuda.is_available()`

 |

Đây là nơi định nghĩa các flag. Flag là cách các bạn truyền thông số vào chương trình để chạy model với nhiều cấu hình khác nhau. Flag được mặc định sẵn các giá trị bằng tham số "default" khi nó được định nghĩa. Ví dụ như ở trên mình thực sự đang chạy chương trình với câu lệnh:

|

1

 |

`python main.py --batch_size=64 --epochs=5 --log-interval=200`

 |

(có nhiều flag khác như không ghi hết để tiết kiệm không gian.)

Tuy nhiên thì các giá trị của các flag này lại trùng với giá trị mặc định nên không ghi cũng không sao. Các bạn có thể thay đổi giá trị các flag bằng việc thay đổi giá trị sau dấu "=". Ví dụ mình muốn chạy model với 10 epoch và batch size 128 thì làm như sau.

|

1

 |

`python main.py --batch_size=128 --epochs=10`

 |

Bài tập: *hãy tìm hiểu ý nghĩa của các flag khác và thử thay đổi chúng xem điều gì xảy ra.*

5\. Phân tích code:

Bây giờ chúng ta sẽ cùng đi sâu vào [main.py](https://github.com/khanhptnk/mnist-pytorch/blob/master/main.py) để học bố cục của một chương trình deep learning. Điều đầu tiên các bạn có thể thấy là chương trình vô cùng ngắn nhờ vào công sức của đội ngũ PyTorch và các thư viện đời trước. Tổng quan, một chương trình deep learning sẽ gồm các phần như sau:

-- Định nghĩa flag.

-- Tải dữ liệu.

-- Định nghĩa model.

-- Vòng lặp train.

-- Vòng lặp test.

a. Định nghĩa flag: đã nói đến ở phần trước.

b. Tải dữ liệu:

|

1

2

3

4

5

6

7

8

9

10

11

12

13

 |

`train_loader` `=` `torch.utils.data.DataLoader(`

`datasets.MNIST(``'./data'``, train``=``True``, download``=``not` `args.no_download_data,`

`transform``=``transforms.Compose([`

`transforms.ToTensor(),`

`transforms.Normalize((``0.1307``,), (``0.3081``,))`

`])),`

`batch_size``=``args.batch_size, shuffle``=``True``,` `*``*``kwargs)`

`test_loader` `=` `torch.utils.data.DataLoader(`

`datasets.MNIST(``'./data'``, train``=``False``, transform``=``transforms.Compose([`

`transforms.ToTensor(),`

`transforms.Normalize((``0.1307``,), (``0.3081``,))`

`])),`

`batch_size``=``args.batch_size, shuffle``=``True``,` `*``*``kwargs)`

 |

Nhiệm vụ của phần này là đọc dữ liệu vào và chia chúng thành các batch để làm input cho model. Các batch này được gộp lại trong một loader ("train_loader" hoặc "test_loader"). Các bạn hình dung mỗi loader là một mảng có nhiều phần tử, mỗi phần tử là một batch. Mỗi batch lại là một mảng có nhiều phần tử, mỗi phần tử là một tấm ảnh hoặc label của tấm ảnh. Mục đích của việc tổ chức dữ liệu như vậy là sao cho các bạn có thể lặp qua từng phần tử của loader để đi qua từng batch một. Hãy nhìn vào method [def train(epoch)](https://github.com/khanhptnk/mnist-pytorch/blob/master/main.py#L93) ta sẽ thấy được vòng lặp này:

|

1

 |

`for` `batch_idx, (data, target)` `in` `enumerate``(train_loader):`

 |

Các biến "data" và "target" lần lượt chứa một batch gồm nhiều tấm ảnh và label tương ứng của chúng.

Các bạn đừng để ý đến những chi tiết phức tạp khác như là cách khai báo loader, các tham số,... Nếu các bạn sử dụng lại code của người khác thì thường là phần tải dữ liệu này được viết sẵn. Để thay đổi bằng dữ liệu của các bạn, không cần viết lại code mà chỉ cần chỉnh lại format của file dữ liệu vào cho đúng với format chuẩn được định nghĩa bởi người viết code đó.

c. Định nghĩa model:

|

1

2

3

4

5

6

7

8

9

10

11

12

13

14

15

16

17

18

19

20

21

22

23

 |

`class` `Net(nn.Module):`

`def` `__init__(``self``):`

`super``(Net,` `self``).__init__()`

`self``.conv1` `=` `nn.Conv2d(``1``,` `10``, kernel_size``=``5``)`

`self``.conv2` `=` `nn.Conv2d(``10``,` `20``, kernel_size``=``5``)`

`self``.conv2_drop` `=` `nn.Dropout2d()`

`self``.fc1` `=` `nn.Linear(``320``,` `50``)`

`self``.fc2` `=` `nn.Linear(``50``,` `10``)`

`def` `forward(``self``, x):`

`x` `=` `F.relu(F.max_pool2d(``self``.conv1(x),` `2``))`

`x` `=` `F.relu(F.max_pool2d(``self``.conv2_drop(``self``.conv2(x)),` `2``))`

`x` `=` `x.view(``-``1``,` `320``)`

`x` `=` `F.relu(``self``.fc1(x))`

`x` `=` `F.dropout(x, training``=``self``.training)`

`x` `=` `F.relu(``self``.fc2(x))`

`return` `F.log_softmax(x)`

`model` `=` `Net()`

`if` `args.cuda:`

`model.cuda()`

`optimizer` `=` `optim.SGD(model.parameters(), lr``=``args.lr, momentum``=``args.momentum)`

 |

Nếu các bạn làm nghiên cứu thì phần lớn các thay đổi sẽ rơi vào phần này. Như mình đã phân tích ở [bài này](http://khanhxnguyen.com/machine-learning-101-supervised-learning-perspective/) thì supervised learning có thể được xem như một dạng tối ưu hàm số. Model có thể được xem như một hàm số fθ(x)fθ(x) với xx là input và θθ là parameter (tham số). Định nghĩa model chính là định nghĩa hàm số này. Với MNIST, model nhận vào một hình ảnh và, thông qua định nghĩa của mình, tính ra độ chắc chắn (là một con số) đối với từng label từ 0 đến 9. Muốn dự đoán cho tấm ảnh, ta chỉ cần lấy label có độ chắn chắn cao nhất.

Model ở đây được định nghĩa bằng một [class Net](https://github.com/khanhptnk/mnist-pytorch/blob/master/main.py#L66). Class này có hai method: *[__init__()](https://github.com/khanhptnk/mnist-pytorch/blob/master/main.py#L66)* và [*forward(x)*](https://github.com/khanhptnk/mnist-pytorch/blob/master/main.py#L66). *__init__()* là constructor của class, nơi định nghĩa các parameter. *forward(x)* là nơi ta định nghĩa các phép tính để tính độ chắc chắn từ input.

Để dễ hiểu hơn, tạm quên đi MNIST, mình giả sử model là một đa thức bậc 2 fθ(x)=ax2+bx+cfθ(x)=ax2+bx+c với tham số θ=(a,b,c)θ=(a,b,c). Chúng ta sẽ định nghĩa model như sau:

|

1

2

3

4

5

6

 |

`class` `Net(nn.Module):`

`def` `__init__(``self``):`

`khai báo parameter là a, b, c.`

`def` `forward(``self``, x):`

`return` `a``*``x^``2` `+` `b``*``x` `+` `c`

 |

Sau khi định nghĩa class của model, ta tạo ra một object model:

|

1

 |

`model` `=` `Net()`

 |

Dựa vào độ chắc chắn của model và đáp án đúng, ta sẽ tính hàm mất mát. Sau đó ta cần một optimizer để tìm ra parameter tối ưu của model sao cho hàm mất mát này đạt cực tiểu.

|

1

 |

`optimizer` `=` `optim.SGD(model.parameters(), lr``=``args.lr, momentum``=``args.momentum)`

 |

Khi tạo object optimizer, bạn phải truyền vào tất cả parameter của model muốn được tối ưu (gọi *model.parameters()*). Các thông số khác như learning rate hoặc là momentum là tùy vào loại optimizer đang được sử dụng. Ở đây, ta đang dùng momentum SGD.

Điều kì diệu của các optimizer này là chúng sẽ tự động dùng backpropagation để tính đạo hàm của hàm mất mát theo từng tham số và thay đổi tham số theo đạo hàm. Chúng ta không cần phải ngồi tự viết công thức đạo hàm, vừa dài vừa dễ sai.

d. Vòng lặp train:

|

1

2

3

4

5

6

7

8

9

10

11

12

13

14

15

 |

`def` `train(epoch):`

`model.train()`

`for` `batch_idx, (data, target)` `in` `enumerate``(train_loader):`

`if` `args.cuda:`

`data, target` `=` `data.cuda(), target.cuda()`

`data, target` `=` `Variable(data), Variable(target)`

`optimizer.zero_grad()`

`output` `=` `model(data)`

`loss` `=` `F.nll_loss(output, target)`

`loss.backward()`

`optimizer.step()`

`if` `batch_idx` `%` `args.log_interval` `=``=` `0``:`

`print``(``'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'``.``format``(`

`epoch, batch_idx` `*` `len``(data),` `len``(train_loader.dataset),`

`100.` `*` `batch_idx` `/` `len``(train_loader), loss.data[``0``]))`

 |

Bố cục chính của vòng lặp train như sau:

|

1

2

3

4

5

6

7

8

 |

`for` `mỗi batch (data, target) (data là x, target là y):`

`1.` `Đưa data vào làm` `input` `cho model và nhận về output:` `-``-` `output` `=` `model(data)` `-``-`

`2.` `Tính hàm mất mát dựa vào output và label đúng:` `-``-` `loss` `=` `F.nll_loss(output, target)` `-``-`

`3.` `Chỉnh lại tham số của model bằng việc gọi optimizer:`

`a. Bỏ hết đạo hàm cũ đi:` `-``-` `optimizer.zero_grad()` `-``-`

`b. Dùng backpropagation tính đạo hàm theo từng tham số:` `-``-` `loss.backward()` `-``-`

`c. Thay đổi tham số dựa vào đạo hàm:` `-``-` `optimizer.step()` `-``-`

`4.` `Thông báo loss trên batch vừa xử lý.`

 |

Các bạn thấy là ở đây model không gọi hàm *model.forward(x)*, mà chỉ đơn giản là *model(data)*. Tuy nhiên, đây chỉ là một mẹo lập trình để rút gọn code mà thôi. Định nghĩa của model phải nằm trong hàm *forward(x) *và các bạn không được đặt tên hàm này khác đi.

e. Vòng lặp test:

|

1

2

3

4

5

6

7

8

9

10

11

12

13

14

15

16

17

18

 |

`def` `test(epoch):`

`model.``eval``()`

`test_loss` `=` `0`

`correct` `=` `0`

`for` `data, target` `in` `test_loader:`

`if` `args.cuda:`

`data, target` `=` `data.cuda(), target.cuda()`

`data, target` `=` `Variable(data, volatile``=``True``), Variable(target)`

`output` `=` `model(data)`

`test_loss` `+``=` `F.nll_loss(output, target).data[``0``]`

`pred` `=` `output.data.``max``(``1``)[``1``]` `# get the index of the max log-probability`

`correct` `+``=` `pred.eq(target.data).cpu().``sum``()`

`test_loss` `=` `test_loss`

`test_loss` `/``=` `len``(test_loader)` `# loss function already averages over batch size`

`print``(``'\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'``.``format``(`

`test_loss, correct,` `len``(test_loader.dataset),`

`100.` `*` `correct` `/` `len``(test_loader.dataset)))`

 |

Vòng lặp test gần như tương tự như vòng lặp train tuy nhiên có một số sự khác biệt sau:\
-- Sau khi train, bạn đã tìm được tham số của model rồi. Bây giờ bạn không cần dùng optimizer để thay đổi tham số nữa mà chỉ việc sử dụng model như một hàm số thông thường, tính output ra từ các input.\
-- Ngoài việc tính hàm mất mát, bạn còn phải tính ra metric bạn thật sự quan tâm (ở đây là accuracy, phần trăm bao nhiêu ví dụ được đoán đúng):

|

1

2

 |

`pred` `=` `output.data.``max``(``1``)[``1``]` `# get the index of the max log-probability`

`correct` `+``=` `pred.eq(target.data).cpu().``sum``()`

 |

Cuối cùng, vì vòng lặp train và test chỉ là vòng lặp cho một epoch mà thôi, chúng ta có một vòng lặp ở cuối chương trình để gọi vòng lặp train và test cho mỗi epoch.

|

1

2

3

 |

`for` `epoch` `in` `range``(``1``, args.epochs` `+` `1``):`

`train(epoch)`

`test(epoch)`

 |

Bài tập: *hãy làm cho model trở nên "deep" bằng cách thêm nhiều layer vào trong định nghĩa của nó. *Gợi ý:* thay đổi hàm *[forward(x)](https://github.com/khanhptnk/mnist-pytorch/blob/master/main.py#L75)*.*

Bài tập: *tìm hiểu ý nghĩa của các hàm *nn.Conv2d()*, *nn.Dropout2d()*, *nn.Linear()* trong hàm *[forward(x)](https://github.com/khanhptnk/mnist-pytorch/blob/master/main.py#L75)*. Gợi ý: search tên chúng ở [PyTorch API reference](http://pytorch.org/api/0.1.3/en/#introduction). *

Bài viết đến đây là hết. Đây là bài dài nhất mình từng viết, nếu có sai sót gì mọi người comment để mình sửa lại. Làm hết bài này chắc cũng bở hơi tai nhưng hy vọng các bạn sẽ có nhiều niềm vui. Happy coding ![🙂](https://s.w.org/images/core/emoji/11/svg/1f642.svg)