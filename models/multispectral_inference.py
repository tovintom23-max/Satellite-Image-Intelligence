"""Phase 4: inference for the trained four-band Bengaluru UNet4 checkpoint.

Input: a 4-band GeoTIFF ordered B04, B03, B02, B08, containing
raw Sentinel-2 surface reflectance DN values (normally 0..10000).
Output: georeferenced class mask, RGB preview, and class statistics.
"""
from pathlib import Path
import uuid

import numpy as np
import rasterio
import torch
from PIL import Image
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = ROOT / 'checkpoints' / 'multispectral' / 'landcover_multispectral.pth'
CLASSES = ('Other', 'Built-up', 'Vegetation', 'Water')
COLORS = np.array([[150,150,150],[220,50,50],[40,160,70],[40,100,220]], dtype=np.uint8)
BANDS = ('B04', 'B03', 'B02', 'B08')

class DoubleConv(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inc,outc,3,padding=1,bias=False),nn.BatchNorm2d(outc),nn.ReLU(inplace=True),
            nn.Conv2d(outc,outc,3,padding=1,bias=False),nn.BatchNorm2d(outc),nn.ReLU(inplace=True))
    def forward(self,x): return self.block(x)

class UNet4(nn.Module):
    def __init__(self,base=16):
        super().__init__()
        self.enc1=DoubleConv(4,base)
        self.enc2=DoubleConv(base,base*2)
        self.enc3=DoubleConv(base*2,base*4)
        self.bottom=DoubleConv(base*4,base*8)
        self.pool=nn.MaxPool2d(2)
        self.up3=nn.ConvTranspose2d(base*8,base*4,2,stride=2)
        self.dec3=DoubleConv(base*8,base*4)
        self.up2=nn.ConvTranspose2d(base*4,base*2,2,stride=2)
        self.dec2=DoubleConv(base*4,base*2)
        self.up1=nn.ConvTranspose2d(base*2,base,2,stride=2)
        self.dec1=DoubleConv(base*2,base)
        self.out=nn.Conv2d(base,4,1)
    def forward(self,x):
        a=self.enc1(x); b=self.enc2(self.pool(a)); c=self.enc3(self.pool(b))
        d=self.bottom(self.pool(c))
        d=self.dec3(torch.cat([self.up3(d),c],dim=1))
        d=self.dec2(torch.cat([self.up2(d),b],dim=1))
        d=self.dec1(torch.cat([self.up1(d),a],dim=1))
        return self.out(d)

def load_model(checkpoint_path=CHECKPOINT):
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    if checkpoint.get('architecture')!='UNet4' or checkpoint.get('num_classes')!=4:
        raise ValueError('Checkpoint architecture/classes do not match UNet4.')
    if tuple(checkpoint.get('input_bands',[]))!=BANDS:
        raise ValueError(f'Expected checkpoint bands {BANDS}.')
    model=UNet4(base=int(checkpoint['base_channels']))
    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    model.eval()
    return model

def predict_multispectral(image_path, output_dir, checkpoint_path=CHECKPOINT, tile_size=256):
    """Predict classes for raw-DN four-band Sentinel-2 GeoTIFF; return output paths/stats.

    Assumes 10 m pixels in a projected CRS for area estimates. Preview PNG is
    for visualization; GeoTIFF retains georeferencing and 255 invalid pixels.
    """
    if tile_size % 8: raise ValueError('tile_size must be divisible by 8')
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    model=load_model(checkpoint_path)
    with rasterio.open(image_path) as src:
        if src.count!=4: raise ValueError(f'Expected 4 bands; found {src.count}')
        descriptions=tuple((d or '').upper() for d in src.descriptions)
        if any(descriptions) and descriptions!=BANDS:
            raise ValueError(f'GeoTIFF band descriptions {descriptions} differ from {BANDS}')
        if src.crs is None: raise ValueError('GeoTIFF must have a CRS')
        if src.crs.is_geographic:
            raise ValueError('Reproject to a metric CRS (e.g. EPSG:32643) before area calculation')
        raw=src.read().astype(np.float32)
        valid=np.all(np.isfinite(raw),axis=0)
        if src.nodata is not None:
            valid &= np.all(raw!=src.nodata,axis=0)
        valid &= np.all(raw>=0,axis=0)
        # Sentinel-2 DN input; do not use this normalization on already scaled [0,1] rasters.
        scaled=np.clip(raw/10000.0,0,1)
        h,w=scaled.shape[1:]
        mask=np.full((h,w),255,dtype=np.uint8)
        with torch.inference_mode():
            for y in range(0,h,tile_size):
                for x in range(0,w,tile_size):
                    patch=scaled[:,y:min(y+tile_size,h),x:min(x+tile_size,w)]
                    ph,pw=patch.shape[1:]
                    tile=np.zeros((4,tile_size,tile_size),dtype=np.float32)
                    tile[:,:ph,:pw]=patch
                    pred=model(torch.from_numpy(tile[None])).argmax(1)[0].numpy()[:ph,:pw].astype(np.uint8)
                    pred[~valid[y:y+ph,x:x+pw]]=255
                    mask[y:y+ph,x:x+pw]=pred
        uid=uuid.uuid4().hex[:10]
        tif=out/f'multispectral_mask_{uid}.tif'
        png=out/f'multispectral_mask_{uid}.png'
        profile=src.profile.copy()
        profile.update(driver='GTiff',count=1,dtype='uint8',nodata=255,compress='deflate')
        with rasterio.open(tif,'w',**profile) as dst:
            dst.write(mask,1)
            dst.set_band_description(1,'Land cover class index')
        preview=np.zeros((h,w,3),dtype=np.uint8)
        for idx,color in enumerate(COLORS):preview[mask==idx]=color
        Image.fromarray(preview).save(png)
        n=int(np.count_nonzero(mask!=255))
        pixel_area=abs(src.transform.a*src.transform.e-src.transform.b*src.transform.d)
        stats={name:{'pixels':int((mask==i).sum()),
                     'percentage':round(100*int((mask==i).sum())/n,2) if n else 0.0,
                     'area_m2':round(int((mask==i).sum())*pixel_area,2)}
               for i,name in enumerate(CLASSES)}
        return {'mask_geotiff':str(tif),'mask_preview':str(png),
                'class_statistics':stats,'valid_pixels':n,'image_width':w,'image_height':h,
                'checkpoint':str(checkpoint_path)}
