"use strict";


var i;
var j;

// 观测序列
//var sentence = '这里输入测试数据';
//var sentence = "全国销量领先的红罐凉茶改成广州恒大";
//var sentence = "这就产生了英国经验主义派的美学以及继起的德美诸国实验美学德国费尔巴哈的以人类学原则为基础的美学法国孔德和泰纳的实证主义美学和自然主义派的小说马克思主义就应时而出日益显示出它的强大威力";

var sentence = "那些不标题就不能使观众懂得内容的美术作品那些可以随便安上一个标题能让观众作各种解释的美术作品那些没有提出任何问题也不犯歪曲现实的大错误的美术作品那些好像任意裁下一个戏剧的片断而又不能使这一片断具备明确的意义的美术品如果作为戏剧中的某一场面的记录或者是作为小说的插图没有什么可以非议的但要是作为一幅独立作战的壁画招贴年画木刻那么在反映生活表达主题上就缺乏独立性不容易发挥美术的战斗作用在中国美术中加一种地方色即成为日本美术它和现实生活相比较和美术同样是讲究集中的表现而不是全盘把生活搬上舞台但它比较缺少语言诸条件的美术在情节描写上是比较完全的不受时间条件限制的就这一点而论美术与戏剧似乎不很相同但是话又说回来不论如何包罗万象的美术与戏剧都只能是生活的某些方面某些片断的选择和重新组织都必须赋予局部以较充实较完全的意义它才说得上是典型的代表的有普遍性的艺术形象中国美术源远流长将中国五千年之艺术瑰宝分门别类集纳成册以继承和发扬祖国的优秀文化传统是美术界和出版界长久以来的夙愿近几年来人们的视野开阔了西方美术潮流源源不断地涌来使画家们陷入了反思也有不少人在迫不及待地改变自己其改变面目之快连自己也将认不得了有经验的老年中年美术教师都有这样的感慨恐怕八十年代有机会进学校受到正规教育的青年虽然学习条件好但创作上未必能赶上他们的大哥大姊有深度因为在今后的创作实践中特别是在美术学校的创作教学中如果不是下很大的决心到生活中去锤炼就很难获得那样的切肤感受我国采用精美的漆器工艺品置于室内作陈设品虽然由来已久但现代磨漆画在继承传统艺术的基础上发挥绘画的形式特点丰富了内涵并给予高一层次的艺术魅力泱泱大度地加入美术画种的行列";


var T = sentence.length;
var N = 4; // 分词使用4种标记：BMES
var V = 65536;

// 初始概率
var PI = new Array(N);
for (i = 0; i < PI.length; i ++) {
	PI[i] = Math.random()
}
//PI[1] = PI[2] = 0;

// 转移概率
var A = new Array(N);
for (i = 0; i < N; i ++) {
	A[i] = new Array(N);
	for (j = 0; j < N; j ++) {
		A[i][j] = Math.random();
	}
}
// 在BMES四种状态之间某些转移是不会发生的
A[0][0]	= A[0][3] = A[1][0] = A[1][3] = A[2][1] = A[2][2] = A[3][1] = A[3][2] = 0;

// 发射概率
var B = new Array(N);
for (i = 0; i < N; i ++) {
	B[i] = new Array(V);
	for (j = 0; j < V; j ++) {
		B[i][j] = Math.random();
	}
}


// 将概率归一化并取log
// 取log的目的是将乘法变为加法，防止数值过小导致下溢
log_normalized(PI);
for (i = 0; i < N; i ++) {
	log_normalized(A[i]);
	log_normalized(B[i]);
}

// 前向概率alpha
var alpha = new Array(T);
for (i = 0; i < alpha.length; i ++) {
	alpha[i] = new Array(N);
}

// 后向概率beta
var beta = new Array(T);
for (i = 0; i < beta.length; i ++) {
	beta[i] = new Array(N);
}

// gamma
var gamma = new Array(T);
for (let t = 0; t < gamma.length; t ++) {
	gamma[t] = new Array(N);
}

// ksi
var ksi = new Array(T-1);
for (let t = 0; t < ksi.length; t ++) {
	ksi[t] = new Array(N);
	for (let i = 0; i < N; i ++) {
		ksi[t][i] = new Array(N);
	}
}

function calc_alpha(PI, A, B, O, alpha) {
	// T = 0
	for (let i = 0; i < N; i ++) {
		alpha[0][i] = PI[i] + B[i][O.charCodeAt(0)];
	}

	let tmp = new Array(N);
	for (let t = 1; t < T; t ++) {
		for (let i = 0; i < N; i ++) {
			for (let j = 0; j < N; j ++) {
				tmp[j] = alpha[t-1][j] + A[j][i];
			}
			alpha[t][i] = log_sum(tmp);
			alpha[t][i] += B[i][O.charCodeAt(t)];
		}
	}
}

function calc_beta(PI, A, B, O, beta) {
	for (let i = 0; i < N; i ++) {
		beta[T-1][i] = 1;
	}

	let tmp = new Array(N);
	for (let t = T-2; t >= 0; t --) {
		for (let i = 0; i < N; i ++) {
			for (let j = 0; j < N; j ++) {
				tmp[j] = A[i][j] + B[j][O.charCodeAt(t+1)] + beta[t+1][j]
			}
			beta[t][i] = log_sum(tmp)
		}
	}
}

function calc_gamma(alpha, beta, gamma) {
	for (let t = 0; t < T; t ++) {
		for (let i = 0; i < N; i ++) {
			gamma[t][i] = alpha[t][i] + beta[t][i];
		}

		let s = log_sum(gamma[t]);
		for (let i = 0; i < N; i ++) {
			gamma[t][i] -= s;
		}
	}
}

function calc_ksi(A, B, O, alpha, beta, ksi) {
	let tmp = new Array(N*N); // 保存某个t下的ksi(i, j)
	let k = 0;

	for (let t = 0; t < T-1; t ++) {
		k = 0;
		for (let i = 0; i < N; i ++) {
			for (let j = 0; j < N; j ++) {
				ksi[t][i][j] = alpha[t][i] + A[i][j] + B[j][O.charCodeAt(t+1)] + beta[t+1][j]
				tmp[k++] = ksi[t][i][j];
			}
		}

		let s = log_sum(tmp);
		for (let i = 0; i < N; i ++) {
			for (let j = 0; j < N; j ++) {
				ksi[t][i][j] -= s;
			}
		}
	}
}

function bw(PI, A, B, O, alpha, beta, gamma, ksi) {
	// 更新初始概率
	for (let i = 0; i < N; i ++) {
		PI[i] = gamma[0][i];
	}

	// 更新转移概率
	let s1 = new Array(T-1);		
	let s2 = new Array(T-1);		
	for (let i = 0; i < N; i ++) {
		for (let j = 0; j < N; j ++) {
			for (let t = 0; t < T-1; t ++) {
				s1[t] = ksi[t][i][j];
				s2[t] = gamma[t][i];
			}
			A[i][j] = log_sum(s1) - log_sum(s2);
		}
	}

	// 更新发射概率
	let s3 = new Array(T);
	let s4 = new Array(T);
	let valid = 0;
	for (let i = 0; i < N; i ++) {
		for (let k = 0; k < V; k ++) {
			valid = 0;
			for (let t = 0; t < T; t ++) {
				if (k == O.charCodeAt(t)) {
					s3[valid++] = gamma[t][i];
				}
				s4[t] = gamma[t][i];
			}

			if (valid == 0) {
				B[i][k] = Math.log(0);
			}
			else {
				B[i][k] = log_sum(s3.slice(0, valid)) - log_sum(s4);
			}
		}
	}
}

function baum_welch() {
	for (let time = 0; time < 200; time ++) {
		calc_alpha(PI, A, B, sentence, alpha);
		calc_beta(PI, A, B, sentence, beta);
		calc_gamma(alpha, beta, gamma);
		calc_ksi(A, B, sentence, alpha, beta, ksi);
		bw(PI, A, B, sentence, alpha, beta, gamma, ksi);

		if ((time+1) % 20 == 0) {
			console.log("training count " + (time+1));
		}

		//console.log("training count: " + time);
	}


	let decode = viterbi(PI, A, B, sentence);
	segment(sentence, decode);
}

function update_once(O) {
	calc_alpha(PI, A, B, O, alpha);
	calc_beta(PI, A, B, O, beta);
	calc_gamma(alpha, beta, gamma);
	calc_ksi(A, B, O, alpha, beta, ksi);
	bw(PI, A, B, O, alpha, beta, gamma, ksi);
}

var delta = new Array(T);
for (let t = 0; t < T; t ++) {
	delta[t] = new Array(N);
	delta[t].fill(0);
}

var pre = new Array(T);
for (let t = 0; t < T; t ++) {
	pre[t] = new Array(N);
	pre[t].fill(0);
}

function viterbi(PI, A, B, O) {
	// T = 0
	for (let i = 0; i < N; i ++) {
		delta[0][i] = PI[i] + B[0][O.charCodeAt(0)];
	}

	// T = 1...T-1
	for (let t = 1; t < T; t ++) {
		for (let i = 0; i < N; i ++) {
			delta[t][i] = delta[t-1][0] + A[0][i];
			pre[t][i] = 0;
			for (let j = 1; j < N; j ++) {
				let jj = delta[t-1][j] + A[j][i];
				if (jj > delta[t][i]) {
					delta[t][i] = jj;
					pre[t][i] = j;
				}
			}
			delta[t][i] += B[i][O.charCodeAt(t)];
		}
	}

	// 回溯获取最大路径
	var decode = new Array(T);
	decode.fill(-1);

	// 先获取最后时刻的最优状态
	let state = 0;
	for (let i = 1; i < N; i ++) {
		if (delta[T-1][i] > delta[T-1][state]) {
			state = i;
		}
	}
	decode[T-1] = state;

	// 回溯
	for (let t = T-2; t >= 0; t --) {
		state = pre[t+1][state];
		decode[t] = state;
	}

	console.log(decode);
	return decode;
}

function segment(O, decode) {
	let i = 0;
	let j = 0;
	while (i < T) { // #B/M/E/S
		if (decode[i] == 0 || decode[i] == 1) {
			j = i + 1;
			while (j < T) {
				if (decode[j] == 2)
					break;
				j ++;
			}
			if (j < T)
				j ++;
			console.log(O.slice(i, j) + "/");
			i = j;
		}
		else if (decode[i] == 3 || decode[i] == 2) {
			console.log(O.slice(i, i+1) + "/");
			i ++;
		}
		else
		{
			console.log("error: " + i + decode[i]);
			i ++;
		}
	}
}

function normalize(a, math_func) {
	let i = 0;
	let j = 0;
	let sum = 0;

	if (math_func == undefined) {
		console.log("math_func is undefined");
	}

	for (i = 0; i < a.length; i++) {
		if (Array.isArray(a[i])) {
			for (j = 0; j < a[i].length; j ++) {
				sum += a[i][j];
			}
		}
		else {
			//console.log("a[" + i + "]: " + a[i]);
			sum += a[i];
		}
	}

	console.log("sum: " + sum);

	// normalize and log
	for (i = 0; i < a.length; i++) {
		if (Array.isArray(a[i])) {
			for (j = 0; j < a[i].length; j ++) {
				a[i][j] /= sum;
				if (math_func != undefined) {
					a[i][j] = math_func(a[i][j]);
				}
			}
		}
		else {
			a[i] /= sum;
			if (math_func != undefined) {
				a[i] = math_func(a[i]);
			}
		}
	}
}

function log_normalized(a) {
	normalize(a, Math.log)
}

// 在将概率值转为LOG值存储后，如果需要将原来的概率值相加，
// 需要先做EXP运算，然后相加，最后再将SUM转为LOG值
function logsumexp(a, b) {
	let vmin = a;
	let vmax = b;

	if (a == -Infinity && b == -Infinity) {
		return -Infinity;
	}

	if (a > b) {
		vmin = b;
		vmax = a;
	}

	if (vmax > vmin + 50) {
		return vmax;
	}
	else {
		return vmax + Math.log(Math.exp(vmin-vmax) + 1.0);
	}
}

function log_sum(a) {
	let sum = a[0];
	for (let i = 1; i < a.length; i ++) {
		sum = logsumexp(sum, a[i]);
	}

	return sum;
}
/*
function log_sum(a) {
	let sum = math.bignumber(0);
	for (let i of a) {
		if ( i < -500 )
			i = -500;

		let b = math.bignumber(i);
		let e = math.exp(b);	
		sum = math.add(sum, e);
	}

	let r = math.log(sum);
	return r.toNumber();
}
*/
