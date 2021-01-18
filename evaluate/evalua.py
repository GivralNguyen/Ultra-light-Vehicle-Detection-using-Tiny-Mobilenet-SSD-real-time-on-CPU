import numpy as np
import os
import sys
import pandas as pd 
from matplotlib import pyplot as plt
from sklearn.cluster import MeanShift, AffinityPropagation
from numpy import unique
from numpy import where
from collections import Counter
from evaluator.calcu_iou import iou_cal, _getArea, _getW_H, _getRatio
from visualize.cluster import cluster_affinityPropagation, cluster_meanShift
import matplotlib.cm as cm

path_miss_box = "miss_wh.txt"

def CalculateAveragePrecision(rec, prec):
        mrec = []
        mrec.append(0)
        [mrec.append(e) for e in rec]
        mrec.append(1)
        mpre = []
        mpre.append(0)
        [mpre.append(e) for e in prec]
        mpre.append(0)
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        ii = []
        for i in range(len(mrec) - 1):
            if mrec[1:][i] != mrec[0:-1][i]:
                ii.append(i + 1)
        ap = 0
        for i in ii:
            ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
        # return [ap, mpre[1:len(mpre)-1], mrec[1:len(mpre)-1], ii]
        return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]
        
def plot_cluster_test(widths, heights, WH_arr, title):
    
    x, y = WH_arr[:, 0], WH_arr[:, 1]
    minx, maxx = np.min(x), np.max(x)
    miny, maxy = np.min(y), np.max(y)
    # hist, _ = np.histogram(x, bins=int(maxx))
    hist, _ = np.histogram(x, bins=len(x))
    print(hist)
    nonzerox = []
    nonzeroy = []
    for i, j in enumerate(list(hist)):
        if j != 0:
            nonzerox.append(i)
            nonzeroy.append(j)
    
    nonzeroy = np.array(nonzeroy).astype(np.float)
    nonzeroy = nonzeroy / np.max(nonzeroy)
    nonzeroy *= maxx

    plt.figure(dpi=150)
    plt.title('fucking scatter')
    plt.plot(x, y, 'r*')
    plt.xlabel('w')
    plt.ylabel('h')

    # plt.hold(True)
    plt.plot(np.array(nonzerox), nonzeroy, 'g--')
    # plt.xlim(0, maxx)
    # plt.ylim(0, maxy)
    plt.show()

def df_plot_cluster_from_size(widths, heights, WH_arr, title):

    """Test AP"""
    model = AffinityPropagation(damping=0.9)
    # fit model and predict clusters
    model.fit(WH_arr)
    yhat = model.predict(WH_arr)
    # retrieve unique clusters
    clusters = unique(yhat)
    """end AP"""

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    # start with a rectangular Figure
    plt.figure(figsize=(8, 8))
    # plt.grid(True)

    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.tick_params(direction='in', top=True, right=True)
    ax_histx = plt.axes(rect_histx)
    ax_histx.tick_params(direction='in', labelbottom=False)
    ax_histy = plt.axes(rect_histy)
    ax_histy.tick_params(direction='in', labelleft=False)

    # the scatter plot:
    # ax_scatter.scatter(x, y)
    colors = cm.rainbow(np.linspace(0, 1, len(clusters)))

    print("Number Class: ", len(clusters))
    for cluster, c in zip(clusters, colors):
        # get row indexes for samples with this cluster
        row_ix = where(yhat == cluster)
        # create scatter of these samples
        ax_scatter.scatter(WH_arr[row_ix, 0], WH_arr[row_ix, 1], color=c)

    # now determine nice limits by hand:
    binwidth = 0.25
    lim = np.ceil(np.abs([widths, heights]).max() / binwidth) * binwidth
    ax_scatter.set_xlim((0, lim))
    ax_scatter.set_ylim((0, lim))
    
    bins = np.arange(0, lim + binwidth, binwidth)
    ax_histx.hist(widths, bins=bins)
    ax_histy.hist(heights, bins=bins, orientation='horizontal')

    ax_histx.set_xlim(ax_scatter.get_xlim())
    ax_histy.set_ylim(ax_scatter.get_ylim())

    ax_scatter.set_xlabel("Value W")
    ax_scatter.set_ylabel("Value H")

    ax_histx.set_ylabel("Number W")
    ax_histy.set_xlabel("Number H")
    point = max(max(widths), max(heights))
    ax_scatter.text(point/1.2, point/1.1, "Number Obj: " + str(len(widths)))
    ax_scatter.text(point/1.2, point/1.2, "Number Class: " + str(len(clusters)))
    ax_scatter.set_title(title)
    plt.show()

def df_plot_bar(df, title, xlabel, ylabel):
    ax = df.plot(kind='hist', alpha=0.7)
    df.plot(kind='kde', ax=ax, secondary_y=True)
    # df.plot.bar(stacked=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    plt.show()

def visual_success_obj(detBoxgts):
    gtw = []
    gth = []
    for box in detBoxgts:
        w ,h = _getW_H(box)
        gtw.append(w)
        gth.append(h)
    df_plot_cluster_from_size(gtw, gth, np.array(detBoxgts), "Cluster Affinity Propagation Detect Success Object ")

def write_missWh_box_img(path_miss_box, array_path):
    with open(path_miss_box, "a") as txt_file:
        for line in array_path:
            # w,h_box, w,h_img
            wh = str(line[0]) + " " + str(line[1])+ " " + str(line[2])+ " " + str(line[3])
            txt_file.write("".join(wh) + "\n")

def write_missWh(path_miss_box, array_path):
    with open(path_miss_box, "a") as txt_file:
        for line in array_path:
            wh = str(line[0]) + " " + str(line[1])
            txt_file.write("".join(wh) + "\n")

def write_info_space(path_miss_box, array_path):
    with open(path_miss_box, "a") as txt_file:
        for line in array_path:
            info = str(line[0]) + " thresMR: " + str(line[1])
            txt_file.write('{}\n'.format(info))

def visual_miss_obj(det, gts, save_miss):
    miss_gtsw = []
    miss_gtsh = []
    miss_wh = []
    for d in det:
        miss_gt = [gt for gt in gts if gt[0] == d]
        for idx in range(len(det[d])):
            if det[d][idx] == 0:
                w, h = _getW_H(miss_gt[idx][3])
                miss_gtsw.append(w)
                miss_gtsh.append(h)
                miss_wh.append([w, h])
    miss_wh = np.array(miss_wh)
    write_missWh(save_miss, miss_wh)
    dict_size = {'W':miss_gtsw, 'H':miss_gtsh}
    df = pd.DataFrame(dict_size)
    # df_plot_bar(df, 'Miss Obj Size', 'Values', 'Number of Obj')
    print("day laf W: ", len(miss_gtsw))
    #plot_cluster_test(miss_gtsw, miss_gtsh, miss_wh, "Cluster Affinity Propagation MissRate Object ")
    # df_plot_cluster_from_size(miss_gtsw, miss_gtsh, miss_wh, "Cluster Affinity Propagation MissRate Object ")
    # cluster_affinityPropagation(miss_wh)
    # cluster_meanShift(miss_wh)

def space_good_MR(dets):
    info_img = []
    for name in dets:
        mr = sum(dets[name])/len(dets[name]) * 100
        for thres_mr in range(0, 110, 10):
            if mr >= (100-thres_mr):
                info_img.append([name, thres_mr])
                break
    return info_img

def metrics_csv(TP, FP, c, npos):
    ret = []
    acc_FP = np.cumsum(FP)
    acc_TP = np.cumsum(TP)
    #npos = len(TP) # 1000 len(TP)
    rec = acc_TP / npos # recall
    prec = np.divide(acc_TP, (acc_FP + acc_TP)) # precision
    [ap, mpre, mrec, ii] = CalculateAveragePrecision(rec, prec)
    r = {
        'class': c,
        'precision': prec,
        'recall': rec,
        'AP': ap,
        'interpolated precision': mpre,
        'interpolated recall': mrec,
        'total positives': npos,
        'total TP': np.sum(TP),
        'total FP': np.sum(FP),
        'num_annos': npos
    }
    return r

def metrics(gtboxs, dtboxs, classes, save_miss=None, IOUThreshold=0.5, save_mr=None, save_dt_right=None):
    """
        c: class
    """
    ret = []
    print("Box GT: ", len(gtboxs))

    # Create list GT value 0
    info_space = Counter([cc[0] for cc in gtboxs])
    for key, val in info_space.items():
        info_space[key] = np.zeros(val)

    dt_right = []
    dt_wrong = []
    print("Box DT: ", len(dtboxs))
    min_obj_mr = sys.float_info.max
    max_obj_mr = sys.float_info.min
    for c in classes:
        gts = []
        [gts.append(g) for g in gtboxs if g[1] == c]
        dects = []
        [dects.append(d) for d in dtboxs if d[1] == c]
        num_annos = len(gts)
        npos = len(gts)
        dects = sorted(dects, key=lambda conf: conf[2], reverse=True)
        TP = np.zeros(len(dects))
        FP = np.zeros(len(dects))
        det = Counter([cc[0] for cc in gts])
        ratio_gt_wh = []
        ratio_dt_wh = []
        wmin, hmin = -1, -1
        wmax, hmax = -1, -1
        detBoxgts = []
        for key, val in det.items():
            det[key] = np.zeros(val)

        for d in range(len(dects)):
            # print('dect %s => %s' % (dects[d][0], dects[d][3],))
            # Find ground truth image
            gt = [gt for gt in gts if gt[0] == dects[d][0]]
            iouMax = sys.float_info.min
            for j in range(len(gt)):
                iou = iou_cal(dects[d][3], gt[j][3])
                ratio_gt_wh.append(_getRatio(gt[j][3]))
                ratio_dt_wh.append(_getRatio(dects[d][3]))
                
                if iou > iouMax:
                    iouMax = iou
                    jmax = j
                else:
                    if _getArea(gt[j][3]) < min_obj_mr:
                        wmin, hmin = _getW_H(gt[j][3])
                        min_obj_mr = _getArea(gt[j][3])
                    if _getArea(gt[j][3]) > max_obj_mr:
                        wmax, hmax = _getW_H(gt[j][3])
                        max_obj_mr = _getArea(gt[j][3])

            if iouMax >= IOUThreshold:
                if det[dects[d][0]][jmax] == 0:
                    TP[d] = 1  # count as true positive
                    det[dects[d][0]][jmax] = 1  # flag as already 'seen'
                    info_space[dects[d][0]][jmax] = 1 #update flag in space GT
                    #width, height box detect
                    #dt_right.append([int(dects[d][3][2]) - int(dects[d][3][0]), int(dects[d][3][3]) - int(dects[d][3][1]), int(dects[d][3][4]), int(dects[d][3][5])])
                    detBoxgts.append(gt[jmax][3])

                    # print("TP")
                else:
                    FP[d] = 1  # count as false positive
                    #dt_wrong.append([int(dects[d][3][2]) - int(dects[d][3][0]), int(dects[d][3][3]) - int(dects[d][3][1]), int(dects[d][3][4]), int(dects[d][3][5])])
                    # print("FP")
            # - A detected "cat" is overlaped with a GT "cat" with IOU >= IOUThreshold.
            else:
                FP[d] = 1  # count as false positive
                #dt_wrong.append([int(dects[d][3][2]) - int(dects[d][3][0]), int(dects[d][3][3]) - int(dects[d][3][1]), int(dects[d][3][4]), int(dects[d][3][5])])

        # visual_success_obj(detBoxgts)
        # if save_miss!=None:
        #     if len(det) != 0 or len(gts) !=0:
        #         visual_miss_obj(det, gts, save_miss)


        acc_FP = np.cumsum(FP)
        acc_TP = np.cumsum(TP)
        rec = acc_TP / npos # recall
        prec = np.divide(acc_TP, (acc_FP + acc_TP)) # precision
        leng_ratio_gt = len(ratio_gt_wh)
        if len(ratio_gt_wh)==0:
            leng_ratio_gt = -1
        leng_ratio_dt = len(ratio_dt_wh)
        if len(ratio_dt_wh)==0:
            leng_ratio_dt = -1
        [ap, mpre, mrec, ii] = CalculateAveragePrecision(rec, prec)
        r = {
            'class': c,
            'precision': prec,
            'recall': rec,
            'AP': ap,
            'interpolated precision': mpre,
            'interpolated recall': mrec,
            'total positives': npos,
            'total TP': np.sum(TP),
            'total FP': np.sum(FP),
            'num_annos': num_annos,
            'MR min obj' :[wmin, hmin],
            'MR max obj': [wmax, hmax],
            'Aver ratio gt': sum(ratio_gt_wh) / leng_ratio_gt,
            'Aver ratio dt': sum(ratio_dt_wh) / leng_ratio_dt
        }
        ret.append(r)
    # print(info_space)
    # space_good_MR(info_space)
    #a = space_good_MR(info_space)
    # write calculator space mr
    if save_mr!=None:
        write_info_space(save_mr, space_good_MR(info_space))
    if save_dt_right!=None:
        write_missWh_box_img(save_dt_right, dt_right)
    if save_miss!=None:
        write_missWh_box_img(save_miss, dt_wrong)
    return ret

def show_metrics(detections, num_img=None):
    # print(r)
    acc_AP = 0
    validClasses = 0
    for metricsClass in detections:
        # Get metric values per each class
        cl = metricsClass['class']
        ap = metricsClass['AP']
        precision = metricsClass['precision']
        recall = metricsClass['recall']
        totalPositives = metricsClass['total positives']
        total_TP = metricsClass['total TP']
        total_FP = metricsClass['total FP']

        fppi = "{0:.2f}".format(total_FP / num_img)
        print('FPPI: %s (%s)' % (fppi, cl))
        mr = "{0:.2f}%".format(100 * (1-(total_TP / metricsClass['num_annos'])))
        print('MR: %s (%s)' % (mr, cl))
        # print("GT Class", total_TP)
        # print("DT Class", total_FP + total_TP)
        if totalPositives > 0:
            validClasses = validClasses + 1
            acc_AP = acc_AP + ap
            prec = ['%.2f' % p for p in precision]
            rec = ['%.2f' % r for r in recall]
            ap_str = "{0:.2f}%".format(ap * 100)
            # ap_str = "{0:.4f}%".format(ap * 100)
            precisi = total_TP/(total_TP+total_FP)
            print('AP: %s (%s)' % (ap_str, cl))
            print('precision: %s (%s)' % (round(precisi,2), cl))
            print('recall: %s (%s)' % (round(recall[-1],2), cl))
            # f.write('\n\nClass: %s' % cl)
            # f.write('\nAP: %s' % ap_str)
            # f.write('\nPrecision: %s' % prec)
            # f.write('\nRecall: %s' % rec)


        # print('MR obj_wh Max:', metricsClass['MR max obj'])
        # print('MR obj_wh min:', metricsClass['MR min obj'])
        # print('Aver ratio gt:', metricsClass['Aver ratio gt'])
        # print('Aver ratio dt:', metricsClass['Aver ratio dt'])
    mAP = acc_AP / validClasses
    mAP_str = "{0:.2f}%".format(mAP * 100)
    print('mAP: %s' % mAP_str)
    return ap
    
if __name__ == "__main__":
    pass