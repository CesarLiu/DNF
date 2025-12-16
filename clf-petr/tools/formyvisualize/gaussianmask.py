#to visualize the gaussian mask
#import matplotlib.pyplot as plt
#gaussian_masks_log = (gaussian_masks + 1).log()
#mask = gaussian_masks_log[img_id, cam_id, :].cpu().numpy()  # Assuming you're using CUDA tensors
#mask = mask.reshape((input_img_h, input_img_w))  # Assuming input_img_h and input_img_w are defined
#plt.gca().invert_yaxis()
#plt.imshow(mask, cmap='jet', interpolation='nearest')
#plt.title(f'Gaussian Mask - Image {img_id}, Camera {cam_id}')
#save_path = f"/home/jguo/PETR/PETR/work_dirs/visualize/image{img_id}_Camera{cam_id}.png"  # 替换为您想要保存的路径plt.savefig(save_path)
#plt.savefig(save_path)
#plt.show()
#then, rotate the image 180 degree to get the correct visualization
##############################################
#for dn

        # #to visualize the reference points and gt boxes and object list
        # import matplotlib.pyplot as plt
        # objforvis = objectlist_bbox[0][:, :2].cpu().numpy()
        # gravity_center_forvis = img_meta['gt_bboxes_3d']._data.gravity_center.cpu().numpy()
        # gravity_center_forvis = gravity_center_forvis[:, :2]
        # outputs_coord_forvis = outputs_coord[0, 0, :, :2].detach().cpu().numpy()
        # plt.figure()
        # plt.scatter(objforvis[:, 0], objforvis[:, 1], c='r', label='objectlist')
        # #plt.scatter(gravity_center_forvis[:, 0], gravity_center_forvis[:, 1], c='b', label='gravity_center')
        # plt.scatter(outputs_coord_forvis[:, 0], outputs_coord_forvis[:, 1], s=20, c='b', label='reference_points_after_transformer')
        # #plt.legend()
        # save_path = f"/home/jguo/PETR/PETR/work_dirs/visualize_for_rp/reference_piont.png"  # 替换为您想要保存的路径plt.savefig(save_path)
        # plt.savefig(save_path)
        # plt.close()

##############################################
#for baseline

        # #to visualize the reference points and gt boxes and object list
        # import matplotlib.pyplot as plt
        # gravity_center_forvis = img_meta['gt_bboxes_3d']._data.gravity_center.cpu().numpy()
        # gravity_center_forvis = gravity_center_forvis[:, :2]
        # all_bbox_preds_forvis = all_bbox_preds[0,0, :, :2].detach().cpu().numpy()
        # plt.figure()
        # #plt.scatter(gravity_center_forvis[:, 0], gravity_center_forvis[:, 1], c='b', label='gravity_center')
        # plt.scatter(all_bbox_preds_forvis[:, 0], all_bbox_preds_forvis[:, 1], s=20,c='g', label='reference_points_after_transformer')
        # #plt.legend()
        # save_path = f"/home/jguo/PETR/PETR/work_dirs/visualize_for_rp/reference_piont_baseline.png"  # 替换为您想要保存的路径plt.savefig(save_path)
        # plt.savefig(save_path)
        # plt.close()

import matplotlib.pyplot as plt

img_path = '/home/jguo/PETR/PETR/work_dirs/visualize_for_rp/reference_piont.png'
mask_path = '/home/jguo/PETR/PETR/work_dirs/visualize_for_rp/reference_piont_baseline.png'

img = plt.imread(img_path)
mask = plt.imread(mask_path)
output_path = f"/home/jguo/PETR/PETR/work_dirs/visualize_for_rp/new1_0.4.png"
# 叠加显示img, mask
plt.imshow(img)
plt.imshow(mask, alpha=0.4)  
plt.savefig(output_path)

plt.show()
